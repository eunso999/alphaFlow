import torch 

def register_sit_block_feature_hooks(sit_model, *, store, retain_grad: bool = False, detach: bool = False):
    """
    sit_model: instance of SiT
    store: dict-like object to store features
    retain_grad: if True, you can read store[name].grad after backward()
    detach: if True, stored features are detached (no grad flow through stored tensor)
    returns: list of hook handles (call .remove() later)
    """
    handles = []
    
    # import ipdb
    # import os
    # rank = int(os.environ.get("RANK", "0"))
    # if rank == 0:
    #     ipdb.set_trace()
    
    for i, blk in enumerate(sit_model.blocks):
        name = f"blocks.{i}"

        def _hook(module, inputs, output, name=name):
            feat = output
            if detach:
                feat = feat.detach()
            if retain_grad and (feat.requires_grad):
                feat.retain_grad()
            store[name] = feat

        handles.append(blk.register_forward_hook(_hook))

    return handles

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

if __name__=="__main__":
    feat_store = {}
    handles = register_sit_block_feature_hooks(model, store=feat_store, retain_grad=True, detach=False)

    out, ctx = model(x, noise_labels=sigma_like, cond=cond, noise_labels_next=sigma_next_like)
    # 이제 feat_store["blocks.0"], ..., feat_store["blocks.K-1"]에 각 block output이 들어있음

    # cleanup
    remove_hooks(handles)