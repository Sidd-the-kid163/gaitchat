from ..models.get_model import get_model as get_gen_model
import clip

def get_model_only(parameters):

    # parameters normally injected by dataset.update_parameters()
    parameters.setdefault("njoints", 25)
    parameters.setdefault("nfeats", 6)
    parameters.setdefault("num_classes", 1)

    clip_model, _ = clip.load("ViT-B/32", device=parameters['device'], jit=False)
    clip.model.convert_weights(clip_model)

    for domain in parameters.get('clip_training', '').split('_'):
        clip_num_layers = parameters.get('clip_layers', 12)

        if domain == 'text':
            clip_model.initialize_parameters()
            clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]

        if domain == 'image':
            clip_model.initialize_parameters()
            clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]

    # freeze CLIP weights
    if parameters.get('clip_training', '') == '':
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    model = get_gen_model(parameters, clip_model)
    return model