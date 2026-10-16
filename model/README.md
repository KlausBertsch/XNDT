# Published model

`best.weights.h5` is the checkpoint used for the reported predictions and
explanations. It was trained with seed 42 on the published training split.
`xndt_normalization.json` contains statistics calculated from that split only.
The manifest hash in `training_summary.json` identifies the pre-publication
export; the later privacy reduction removed only fields unused by the model.

SHA-256:

- `best.weights.h5`: `5851f94341afcd35e8e365079224ac0f01703086fbf14c0c275ae40c724fff82`
- `xndt_normalization.json`: `8e4cff6d12fbeb8bf9677a0fdecb20084e29e94623cc5fecc8d649e13b645b8a`
