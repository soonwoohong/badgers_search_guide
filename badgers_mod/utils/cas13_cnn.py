"""Utility script to enable the use of the Cas13a predictive models from the ADAPT publication (https://www.nature.com/articles/s41587-022-01213-5)"""
import os
import adapt
from adapt.utils import predict_activity
import tensorflow as tf

dir_path = adapt.utils.version.get_project_path()
cla_path_all = os.path.join(dir_path, 'models', 'classify', 
                                        'cas13a')
reg_path_all = os.path.join(dir_path, 'models', 'regress',
                                        'cas13a')
cla_version = adapt.utils.version.get_latest_model_version(cla_path_all)
reg_version = adapt.utils.version.get_latest_model_version(reg_path_all)
cla_path = os.path.join(cla_path_all, cla_version)
reg_path = os.path.join(reg_path_all, reg_version)
pred = predict_activity.Predictor(cla_path, reg_path)

target_len = 48
guide_len = 28
#pad_len = tf.constant((target_len - guide_len) / 2)
pad_len = (target_len-guide_len)//2

def _to_tensor(x, dtype=None):
    # Accept list/tuple/np/tf and return tf.Tensor
    if isinstance(x, (list, tuple)):
        x = tf.convert_to_tensor(x)
    elif not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x)
    if dtype is not None and x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

@tf.function
def run_full_model(gen_guide, target_set, model_type = 'both'):
    """
    Modified version:
        gen_guide : [B, 28, 4] one-hot (float16/float32) or list of [28,4]
        target_set: [T, 48, 4] one-hot (float16/float32) or list of [48,4]
    Returns:
      - if model_type=='both'     : (pred_perf [B,T], classify_perf [B,T])
      - if model_type=='regress'  : pred_perf [B,T]
      - if model_type=='classify' : classify_perf [B,T]


    Original:
    Function to run the both the predictive and regression model from the ADAPT publication (https://www.nature.com/articles/s41587-022-01213-5)
    Args:
        gen_guide: List of guide sequences to be evaluated
        target_set: List of target sequences to be evaluate each guide sequence against
        model_type: Type of model to run. Run the regression model, classification model or both.
    Returns:
        pred_perf_list: List of predicted performance values for each guide-target pair
        classify_perf_list: List of predicted classification values for each guide-target pair
    """

    assert model_type in ['regress', 'both', 'classify']

    # checking
    tf.debugging.assert_rank_in(gen_guide, [2, 3])
    tf.debugging.assert_rank_in(target_set, [2, 3])

    # Normalize inputs to rank-3 tensors
    # (If rank-2 per-seq, stacking above already handled by caller)
    gen_guide = _to_tensor(gen_guide)
    target_set = _to_tensor(target_set, dtype=gen_guide.dtype)  # keep same dtype (fp16 if mixed precision)

    # Ensure shapes are [B, 28, 4] and [T, 48, 4]
    gen_guide = tf.ensure_shape(gen_guide, [None, guide_len, 4])
    target_set = tf.ensure_shape(target_set, [None, target_len, 4])

    B = tf.shape(gen_guide)[0] # batch size
    T = tf.shape(target_set)[0] # number of target

    # Pad guides to target length: [B, 48, 4] outside the for-loop
    guides_padded = tf.pad(gen_guide, paddings=[[0, 0], [pad_len, pad_len], [0, 0]])

    # Cross-join guides and targets on-device:
    # guides_x:  [B, T, 48, 4]
    # targets_x: [B, T, 48, 4]
    guides_x = tf.broadcast_to(guides_padded[:, tf.newaxis, :, :], [B, T, target_len, 4])
    targets_x = tf.broadcast_to(target_set[tf.newaxis, :, :, :], [B, T, target_len, 4])

    # Prepare the guide-target arrays for the pred model
    # Concatenate along channels -> [B, T, 48, 8] then flatten batch for the model
    pred_in = tf.concat([targets_x, guides_x], axis=-1)             # [B, T, 48, 8]
    pred_in = tf.reshape(pred_in, [B * T, target_len, 8])           # [B*T, 48, 8]

    outputs = []

    if model_type in ('regress', 'both'):
        reg = pred.regression_model(pred_in, training=False)  # [B*T, 1] or [B*T]
        reg = tf.squeeze(reg, axis=-1) if reg.shape.rank == 2 else reg
        reg = tf.reshape(reg, [B, T])  # -> [B, T]
        outputs.append(reg)

    if model_type in ('classify', 'both'):
        cla = pred.classification_model(pred_in, training=False)  # [B*T, 1] or [B*T]
        cla = tf.squeeze(cla, axis=-1) if cla.shape.rank == 2 else cla
        cla = tf.reshape(cla, [B, T])  # -> [B, T]
        outputs.append(cla)

    if model_type == 'both':
        return outputs[0], outputs[1]
    else:
        return outputs[0]

'''
    original
    
    # Prepare the guide-target arrays for the pred model
    #pred_input_list  = []

    for guide in gen_guide:
        #print(f"Guide Length: {len(guide)}")
        for target in target_set:
            #print(f"Target Length: {len(target)}") 
            gen_guide_padded = tf.pad(guide, [[pad_len, pad_len], [0, 0]])
            pred_input_list.append(tf.concat([target, gen_guide_padded], axis=1))

    regression_output = pred.regression_model.call(pred_input_list, training = False)
    #pred_perf_list = tf.split(tf.reshape(regression_output, [len(target_set) * len(gen_guide)]), len(gen_guide))
    pred_perf_list = tf.reshape(regression_output, [len(target_set) * len(gen_guide)]), len(gen_guide)

    classifier_output = pred.classification_model.call(pred_input_list, training = False)
    #classify_perf_list = tf.split(tf.reshape(classifier_output, [len(target_set) * len(gen_guide)]), len(gen_guide))
    classify_perf_list = tf.reshape(classifier_output, [len(target_set) * len(gen_guide)]), len(gen_guide)

    if(model_type == 'both'):
        return pred_perf_list, classify_perf_list
    elif(model_type == 'regress'):
        return pred_perf_list
    elif(model_type == 'classify'):
        return classify_perf_list
'''