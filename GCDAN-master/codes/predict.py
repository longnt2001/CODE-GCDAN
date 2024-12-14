import torch
import json
from model import TrajTransformer
from utils import RnnParameterData

# Load the model with the saved checkpoint
def load_model(parameters, checkpoint_path):
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint keys:", checkpoint.keys())
    print("Shape of lut_loc.weight:", checkpoint['model.src_embed.0.lut_loc.weight'].shape)

    # Fix parameters.loc_size to match the checkpoint size
    parameters.loc_size = checkpoint['model.src_embed.0.lut_loc.weight'].shape[1]

    # Adjust checkpoint weights to match the model if necessary
    if checkpoint['model.src_embed.0.lut_loc.weight'].shape[1] != parameters.loc_size:
        print("Adjusting src_embed.0.lut_loc.weight")
        weight_diff = parameters.loc_size - checkpoint['model.src_embed.0.lut_loc.weight'].shape[1]
        if weight_diff > 0:
            additional_weights = torch.zeros((512, weight_diff))
            checkpoint['model.src_embed.0.lut_loc.weight'] = torch.cat(
                [checkpoint['model.src_embed.0.lut_loc.weight'], additional_weights], dim=1
            )
        else:
            checkpoint['model.src_embed.0.lut_loc.weight'] = checkpoint['model.src_embed.0.lut_loc.weight'][:, :parameters.loc_size]

    if checkpoint['model.tgt_embed.0.lut_loc.weight'].shape[1] != parameters.loc_size:
        print("Adjusting tgt_embed.0.lut_loc.weight")
        weight_diff = parameters.loc_size - checkpoint['model.tgt_embed.0.lut_loc.weight'].shape[1]
        if weight_diff > 0:
            additional_weights = torch.zeros((512, weight_diff))
            checkpoint['model.tgt_embed.0.lut_loc.weight'] = torch.cat(
                [checkpoint['model.tgt_embed.0.lut_loc.weight'], additional_weights], dim=1
            )
        else:
            checkpoint['model.tgt_embed.0.lut_loc.weight'] = checkpoint['model.tgt_embed.0.lut_loc.weight'][:, :parameters.loc_size]

    if checkpoint['model.generator.proj.weight'].shape[0] != parameters.loc_size:
        print("Adjusting generator.proj.weight")
        weight_diff = parameters.loc_size - checkpoint['model.generator.proj.weight'].shape[0]
        if weight_diff > 0:
            additional_weights = torch.zeros((weight_diff, 544))
            checkpoint['model.generator.proj.weight'] = torch.cat(
                [checkpoint['model.generator.proj.weight'], additional_weights], dim=0
            )
        else:
            checkpoint['model.generator.proj.weight'] = checkpoint['model.generator.proj.weight'][:parameters.loc_size, :]

    if checkpoint['model.generator.proj.bias'].shape[0] != parameters.loc_size:
        print("Adjusting generator.proj.bias")
        bias_diff = parameters.loc_size - checkpoint['model.generator.proj.bias'].shape[0]
        if bias_diff > 0:
            additional_bias = torch.zeros(bias_diff)
            checkpoint['model.generator.proj.bias'] = torch.cat(
                [checkpoint['model.generator.proj.bias'], additional_bias], dim=0
            )
        else:
            checkpoint['model.generator.proj.bias'] = checkpoint['model.generator.proj.bias'][:parameters.loc_size]

    model = TrajTransformer(parameters=parameters)
    model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
    model.eval()
    return model

# Prepare dataset for inference
def prepare_input_data(dataset, user_id, mode="test", max_len=20):
    """
    Prepare input data for a specific user from the dataset.
    Args:
        dataset: Loaded dataset.
        user_id: The user ID to extract data for.
        mode: "train" or "test" to select the corresponding split.
        max_len: Maximum sequence length for input.
    Returns:
        Prepared input tensors.
    """
    # Assuming data_neural contains the required session data
    user_data = dataset['data_neural'][user_id]
    sessions = user_data['sessions'] if 'sessions' in user_data else user_data
    indices = user_data[mode] if mode in user_data else range(len(sessions))

    print("Sessions content:", sessions)  # Debugging: print sessions content

    src_loc, src_st, src_ed = [], [], []
    for i in indices:
        if isinstance(sessions, dict):
            session = sessions[str(i)]  # Access by key if sessions is a dict
        else:
            session = sessions[i]  # Access by index if sessions is a list

        loc = [s[0] + 1 for s in session]
        st = [s[1] + 1 for s in session]
        ed = [s[2] + 1 for s in session]

        # Padding to max_len
        if len(loc) < max_len:
            pad_len = max_len - len(loc)
            loc.extend([0] * pad_len)
            st.extend([0] * pad_len)
            ed.extend([0] * pad_len)
        
        src_loc.append(loc[:max_len])
        src_st.append(st[:max_len])
        src_ed.append(ed[:max_len])

    src_loc = torch.LongTensor(src_loc)
    src_st = torch.LongTensor(src_st)
    src_ed = torch.LongTensor(src_ed)
    return src_loc, src_st, src_ed

# Inference function
def predict(model, input_data, parameters):
    """
    Perform prediction using the trained model.
    Args:
        model: Trained model (TrajTransformer).
        input_data: Input data prepared in the format expected by the model.
        parameters: Parameters object for input embedding.
    Returns:
        Prediction results.
    """
    src_loc, src_st, src_ed = input_data

    with torch.no_grad():
        # Convert tensors to float for compatibility
        src_loc = src_loc.float()
        src_st = src_st.float()
        src_ed = src_ed.float()

        src_mask = (src_loc != 0).unsqueeze(-2)

        # Dummy target tensors for inference
        tgt_loc = torch.zeros_like(src_loc).float()
        tgt_st = torch.zeros_like(src_st).float()
        tgt_ed = torch.zeros_like(src_ed).float()
        tgt_mask = (tgt_loc != 0).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.triu(torch.ones(tgt_loc.size(-1), tgt_loc.size(-1)), diagonal=1).bool()

        # Model forward pass
        output = model(src_loc, src_st, src_ed, tgt_loc, tgt_st, tgt_ed, [20], [1])
        return output

# Main function
if __name__ == "__main__":
    # Load dataset
    dataset_path = "./data/anonymized_wifi.json"  # Path to your JSON dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Print dataset keys for debugging
    print("Dataset keys:", dataset.keys())
    print("Type of data_neural:", type(dataset['data_neural']))

    # If data_neural is a dictionary, print its keys
    if isinstance(dataset['data_neural'], dict):
        print("Keys in data_neural:", dataset['data_neural'].keys())

    # Initialize parameters
    parameters = RnnParameterData(
        loc_emb_size=512, uid_emb_size=128, tim_emb_size=16,
        dropout_p=0.1, data_name="wifi", lr=5e-5,
        lr_step=3, lr_decay=0.1, L2=1e-5, optim="Adam",
        clip=5.0, epoch_max=5, data_path="./data/", save_path="./results/"
    )

    # Manually add all required attributes matching the checkpoint
    parameters.tim_size = 48     # Adjusted to match checkpoint
    parameters.uid_size = 886    # Adjusted to match checkpoint
    parameters.loc_emb_size = 512
    parameters.uid_emb_size = 128
    parameters.tim_emb_size = 16
    parameters.dropout_p = 0.1  # Adding dropout_p explicitly

    # Load model
    checkpoint_path = "./results/checkpoint/ep_4.m"  # Update this to your checkpoint
    model = load_model(parameters, checkpoint_path)

    # Select a valid user ID from dataset
    user_id = next(iter(dataset['data_neural'].keys()))  # Get the first key dynamically

    # Prepare input data
    input_data = prepare_input_data(dataset, user_id, mode="test")

    # Run inference
    predictions = predict(model, input_data, parameters)
    print("Predictions:", predictions)
