import { useEffect, useRef } from "react";

export function useSectionSocket(
  sectionId: string,
  userid: number,
  active: boolean,
  onMessage: (text: string) => void,
  onReady?: () => void,
  onResponse?: () => void
) {
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!active) return;  // don't connect if section not active

    const wsUrl = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";
    const ws = new WebSocket(`${wsUrl}/ws/${sectionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(`USERID:${userid}`);
    };

    ws.onmessage = (e) => {
      if (e.data === "TEXT:ready") {
        onReady?.();
        return;
      }
      onResponse?.();
      onMessage(e.data);
    };

    ws.onerror = () => {};
    ws.onclose = (e) => console.log("WS closed", sectionId, "code:", e.code);
    return () => ws.close();
  }, [sectionId, userid, active]);

  function send(text: string) {
    wsRef.current?.send(text);
  }

  return { send };
}