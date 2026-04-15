import pyodbc
import os
import sys
import numpy as np
import torch
import time
import io
from datetime import datetime
from pydantic import BaseModel
import random
from os.path import join as pjoin
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from fastapi.staticfiles import StaticFiles
import csv
import base64

# ── Database connection ───────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()
DB_SERVER = os.getenv("DB_SERVER", "semrob.database.windows.net")
DB_DATABASE = os.getenv("DB_DATABASE", "robotic-gifs")
DB_USERNAME = os.getenv("DB_USERNAME", "admin163")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")
DB_TRUST_SERVER_CERTIFICATE = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "Yes")
DB_CONNECTION_TIMEOUT = os.getenv("DB_CONNECTION_TIMEOUT", "30")

datatables = ['cmu','jump_handsup','jump_vertical','run','sit','walk']

# -------- standalone quaternion helpers --------
def qinv(q):
    # q shape: (..., 4), format [w, x, y, z]
    mask = torch.ones_like(q)
    mask[..., 1:] = -1
    return q * mask

def qrot(q, v):
    # q shape: (..., 4), v shape: (..., 3)
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError("q must end with 4 and v must end with 3")
    if q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q and v batch dims must match")

    original_shape = v.shape
    q_flat = q.contiguous().view(-1, 4)
    v_flat = v.contiguous().view(-1, 3)

    qvec = q_flat[:, 1:]
    uv = torch.cross(qvec, v_flat, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    out = v_flat + 2.0 * (q_flat[:, :1] * uv + uuv)
    return out.view(original_shape)

# -------- recovery functions --------
def recover_root_rot_pos(data):
    # data: (B, T, D)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel, device=data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num=22):
    # data: (B, T, D), D usually 263 for joints_num=22
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions

def connect_with_retry(max_attempts=5, delay=1):
    for attempt in range(max_attempts):
        try:
            conn = pyodbc.connect(
                f"Driver={{ODBC Driver 18 for SQL Server}};Server=tcp:{DB_SERVER},{DB_PORT};Database={DB_DATABASE};Uid={DB_USERNAME};Pwd={DB_PASSWORD};Encrypt={DB_ENCRYPT};TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE};Connection Timeout={DB_CONNECTION_TIMEOUT};"
            )
            print(f"DB connected on attempt {attempt + 1}")
            return conn
        except pyodbc.Error as e:
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to DB after multiple attempts")

def change_tonpyjoints(filename):
    for table in datatables:
        if table == 'cmu':
            query = f"SELECT npy_data FROM {table} WHERE filename = ?"
        else:
            query = f"SELECT npy_data FROM {table} WHERE origname = ?"

        cursor.execute(query, (filename,))
        row = cursor.fetchone()  # only fetch one row
        if row:
            return np.load(io.BytesIO(row[0]))  # load binary data into numpy array

def change_tonpy(origname):
    filename = os.path.basename(origname)
    query = f"SELECT upperstatic FROM cmu WHERE filename = ?"
    cursor.execute(query, (filename,))
    row = cursor.fetchone()  # only fetch one row
    if row:
        return np.load(io.BytesIO(row[0])) 

def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
#     matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    #ax = p3.Axes3D(fig) can silently fail
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        ax.cla()
        # Reset limits after cla()
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.grid(False)
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
#         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, fps=fps)
    plt.close()

def render_motion(text, dataset_id):
    if dataset_id == 1:
        kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        save_path = pjoin("./input/", text + '1.mp4')
        x = torch.from_numpy(change_tonpy(text)).float().unsqueeze(0)           # (1, T, D)
        joints = recover_from_ric(x, 22)                        # (1, T, 22, 3)

        joints_np = joints.squeeze(0).cpu().numpy()  
        plot_3d_motion(save_path, kinematic_chain, joints_np, title="Upperstatic", fps=20, radius=4)
        return save_path
    else:
        kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        save_path = pjoin("./input/", text + '.mp4')
        plot_3d_motion(save_path, kinematic_chain, change_tonpyjoints(text), title="Normal", fps=20, radius=4)
        return save_path

conn = connect_with_retry()
cursor = conn.cursor()

# ── Table names ───────────────────────────────────────────────────────────────

datatables = ['cmu', 'jump_handsup', 'jump_vertical', 'run', 'sit', 'walk']
test       = ['jump_handsup', 'jump_vertical', 'run', 'sit', 'walk']

# ── Load datasets from DB on startup ─────────────────────────────────────────

cmu          = []
jump_handsup = []
jump_vertical= []
run          = []
sit          = []
walk         = []

cursor.execute("SELECT outside, filename, sequence_labels, frame_labels, learning_data FROM dbo.cmu;")
for outside, filename, sequence_labels, frame_labels, learning_data in cursor.fetchall():
    cmu.append([outside, filename, sequence_labels or "no summary", frame_labels or "unspecified"])

for x in test:
    cursor.execute(f"SELECT origname FROM dbo.{x};")
    for origname in cursor.fetchall():
        if x == 'jump_handsup':
            jump_handsup.append([origname[0]])
        elif x == 'jump_vertical':
            jump_vertical.append([origname[0]])
        elif x == 'run':
            run.append([origname[0]])
        elif x == 'sit':
            sit.append([origname[0]])
        elif x == 'walk':
            walk.append([origname[0]])

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this:
from fastapi.responses import FileResponse
from fastapi import Response

@app.get("/video")
async def get_video(path: str):
    clean_path = path.split("?")[0]  # strip ?t=timestamp
    if os.path.exists(clean_path):
        media_type = "image/gif" if clean_path.endswith(".gif") else "video/mp4"
        return FileResponse(clean_path, media_type=media_type)
    return Response(status_code=404)

@app.get("/render")
async def render_input(filename: str, dataset_id: int):
    if dataset_id == 1:
        path = f"./input/{filename}1.mp4"
    else:
        path = f"./input/{filename}.mp4"
    if os.path.exists(path):
        return {"path": path}
    # File doesn't exist, render it
    path = render_motion(filename, dataset_id)
    if path and os.path.exists(path):
        return {"path": path}
    return Response(status_code=404)

# ── /datasets endpoint ────────────────────────────────────────────────────────

@app.get("/datasets")
async def get_datasets():
    return {
        "cmu": [
            {
                "outside":          row[0],
                "origname":         row[1],
                "sequence_labels":   row[2],
                "frame_labels": row[3],
            }
            for row in cmu
        ],
        "upperstatic": [
            {
                "outside":          row[0],
                "origname":         row[1],
                "sequence_labels":   row[2],
                "frame_labels": row[3],
            }
            for row in cmu
        ],
        "jump_handsup":  [{"origname": row[0]} for row in jump_handsup],
        "jump_vertical": [{"origname": row[0]} for row in jump_vertical],
        "run":           [{"origname": row[0]} for row in run],
        "sit":           [{"origname": row[0]} for row in sit],
        "walk":          [{"origname": row[0]} for row in walk],
    }

# ── Process management ────────────────────────────────────────────────────────

SCRIPTS = {
    #"01": {"env": "motiondiffuse", "script": "script_01/motiondiffuse/plugplay.py"},
    "02": {"env": "mgpt",     "script": "script_02/motiongpt3/plugplay.py"},
    #"03": {"env": "mdm",           "script": "script_03/mdm/plugplay.py"},
    #"04": {"env": "T2M-GPT",        "script": "script_04/t2mgpt/plugplay.py"},
    #"05": {"env": "momask",        "script": "script_05/momask/plugplay.py"},
    #"06": {"env": "motionclip",    "script": "script_06/motionclip/plugplay.py"},
    #"07": {"env": "motionagent",   "script": "script_07/motionagent/demo.py"},
}
processes: dict[str, asyncio.subprocess.Process] = {}

async def start_process(section_id: str):
    config = SCRIPTS[section_id]
    script_abs = os.path.abspath(config["script"])
    script_dir = os.path.dirname(script_abs)
    proc = await asyncio.create_subprocess_exec(
        "conda", "run", "--no-capture-output", "-n", config["env"],
        "python", "-u", script_abs,  # use absolute path
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=script_dir,
    )
    processes[section_id] = proc
    print(f"Started section {section_id} with env {config['env']} PID {proc.pid}")
    async def read_stderr():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                print(f"[{section_id} stderr] {line.decode().rstrip()}")
        except Exception:
            pass
    asyncio.create_task(read_stderr())

    return proc

async def watch_process(section_id: str):
    await asyncio.sleep(5)  # wait for model to load first
    while True:
        proc = processes.get(section_id)
        if proc is not None:
            await proc.wait()  # block until process exits
            print(f"Section {section_id} exited with code {proc.returncode}, restarting...")
            await asyncio.sleep(5)  # brief delay before restart
            await start_process(section_id)
        else:
            await asyncio.sleep(5)

@app.get("/sections")
async def get_sections():
    return {"active": list(SCRIPTS.keys())}

@app.on_event("startup")
async def startup():
    for section_id in SCRIPTS:
        await start_process(section_id)
        asyncio.create_task(watch_process(section_id))  # re-enable
        print(f"Started section {section_id}")

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{section_id}")
async def websocket_endpoint(websocket: WebSocket, section_id: str):
    print(f"WS attempt: {section_id}")
    await websocket.accept()
    print(f"WS accepted: {section_id}")

    if section_id not in processes:
        await websocket.send_text("ERROR: unknown section")
        await websocket.close()
        return

    proc = processes[section_id]

    async def read_output():
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode().rstrip()
                if decoded.startswith("TEXT:") or decoded.startswith("FILE:mp4:") or decoded.startswith("FILE:gif:"): #FIXME
                    await websocket.send_text(decoded)
        except Exception:
            pass

    output_task = asyncio.create_task(read_output())

    try:
        while True:
            data = await websocket.receive_text()

            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.write((data + "\n").encode())
                await proc.stdin.drain()
            #else: silently do nothing

    except WebSocketDisconnect:
        pass
    finally:
        output_task.cancel()

#exit strategy
class FeedbackRequest(BaseModel):
    userid: int
    rating: str
    comment: str = ""

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    for i in range(1, 8):
        history_var = f'history{i}'
        check_var = f'usercheck{i}'
        cursor.execute(f"""
            SELECT *
            FROM dbo.{history_var}
            ORDER BY created_at DESC
        """)
        last_20 = cursor.fetchall()
        extra_col_name = "comments"
        extra_value_first_row = feedback.comment
        selected_idx = [0, 1, 3, 4, 5, 6, 8]   # choose the column positions you want
        columns = [col[0] for col in cursor.description] + [extra_col_name]
        selected_headers = [columns[i] for i in selected_idx]

        with open(f"{history_var}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(selected_headers)

            for r_idx, row in enumerate(last_20):
                base = [row[i] for i in selected_idx]
                extra = extra_value_first_row if r_idx == 0 else ""
                writer.writerow(base + [extra])
        for x in last_20:
            cursor.execute(f"""
            INSERT INTO dbo.{check_var}
            (query, comments, outputtext, outputmotion, userid, task, inputmotion, created_at, history, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (x[0], feedback.comment, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]))
        #cursor.execute(f"""DELETE FROM dbo.{history_var} WHERE userid = ?""", (feedback.userid,))
        cursor.execute(f"DELETE FROM dbo.{history_var}")
    conn.commit()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)