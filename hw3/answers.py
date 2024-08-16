r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=256,
        seq_len=64,
        h_dim=512,
        n_layers=3,
        dropout=0.15,
        learn_rate=0.001,
        lr_sched_factor=0.09,
        lr_sched_patience=4,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "In the begining"
    temperature = 0.01
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

Splitting the corpus into sequences for training is crucial for managing memory, and enabling efficient batch processing. This allows for better generalization and faster training because it can learn shorter patterns which wont be the case for one long text. This method also reflects real-world applications, where models often process shorter text sequences.
"""

part1_q2 = r"""
**Your answer:**

RNN models has memory that is implemented using the hidden state. Each step in training the hidden state is updated. The model is a next char predictor that uses this hidden state and the last char as input, because of it's recurrent nature, the output sequence can be as long as we wish it to be, although it might not be coherent text at some point since the model was trained on finite sized sequences.
"""

part1_q3 = r"""
**Your answer:**


It might seem that the order of the batches is not important but recall that the input dataset (the corpus) is a coherent text that has a lot of contextuality and long distance correlations (e.g the first sentence of the corpus might be necessary to understand the last sentence). If we shuffle the batches we lose contextuality and we have no reason to expect the model to learn these long distance correlations. 
"""

part1_q4 = r"""
**Your answer:**

1. When we lower the temperature compared to 1.0 we essentialy amplify the probablities- lower values are suppressed and larger values are amplified. This way it makes decision making easier because the probability distrubtion is non-uniform, the lower the temperature the more non-uniform the distrubtion becomes.

2. When the temperature is higher the distribution is closer to uniform adding more randomness to the output. Outputs will be more varied as for uniform distribution it is equally likely to get diffrenet ouputs. As a result the outputs will be much less coherent from one another and much more diverse.

3. When the temperature is lower the distribution is very non-uniform like discussed in Q4.1, this means that the output stability much higher- the model will generate very similar results to one another. This will reduce creatavity, the outputs will be very close to the text given in training and we will notice very reptative behavior.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = "https://drive.google.com/uc?export=download&id=1MkYZikr9h3eplpjaFCf6Ac_0ZuHIu1RB.zip"

def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers

def part3_gan_hyperparams():
        return {
                    'batch_size': 64,                    # Reduced batch size for potentially more stable training
                    'z_dim': 128,                        # Slightly smaller latent dimension
                    'discriminator_optimizer': {
                        'type': 'Adam',                  # Optimizer type for Discriminator
                        'lr': 0.0002,                    # Reduced learning rate for stability
                        'betas': (0.5, 0.999)            # Beta parameters for Adam
                    },
                    'generator_optimizer': {
                        'type': 'Adam',                  # Optimizer type for Generator
                        'lr': 0.00025,                    # Reduced learning rate for stability
                        'betas': (0.5, 0.999)            # Beta parameters for Adam
                    },
                    'data_label': 1,                     # Label for real data
                    'label_noise': 0.15                  # Increased noise added to labels
                }


part2_q1 = r"""
**Your answer:**


Gradients are maintained during GAN training when updating the Generator and Discriminator to adjust their parameters based on their respective losses to improve their preformance. During sampling or evaluation, gradients are discarded because no parameter updates are needed, allowing for more efficient computation and avoiding unnecessary memory use. This approach ensures that the training process focuses on improving the model, while sampling remains efficient and separate from the training phase.
"""

part2_q2 = r"""
**Your answer:**

1. Relying solely on the Generator loss being below a certain threshold to stop GAN training is not ideal, as it doesn't reflect the balance between the Generator and Discriminator. The Generator might have a low loss while still producing poor or repetitive images. It's essential to also track the Discriminator's performance and evaluate the quality of generated images using metrics that combine both Generator's loss and Discriminator's loss as well.

2. If the Discriminator loss stays constant while the Generator loss decreases, it means the Generator is getting better at creating realistic samples. The Discriminator is struggling to improve its accuracy, possibly because it has reached its limit or is overfitting. This situation suggests the Generator is making progress, but the Discriminator's ability to differentiate is stagnating.

"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = "https://drive.google.com/uc?export=download&id=1MkYZikr9h3eplpjaFCf6Ac_0ZuHIu1RB"


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim = 256, 
        num_heads = 4,
        num_layers = 3,
        hidden_dim = 128,
        window_size = 32,
        droupout = 0.2,
        dropout = 0.2,
        lr=0.0001
    )
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

Stacking encoder layers with sliding-window attention extends the context in the final layer, similar to how deeper CNNs capture larger image areas. Each layer focuses on a local window, and stacking layers combines these local views into a broader understanding of the entire input sequence. Essentialy, for every layer, the receptive field increases and we get a more broad context.
"""

part3_q2 = r"""
**Your answer:**

We can use dilated windows where the attention is computed over a fixed-size window but with gaps between the elements. For example, instead of attending to every position within a window, attend to every second or third position in a bigger window, thus maintaining the same amount of attention positions so the complexity remains the same. This approach increases the receptive field size without increasing the complexity significantly.
"""


part4_q1 = r"""
**Your answer:**

The results show that fine-tuning a pre-trained model significantly outperforms using a model trained from scratch. This is likely because the pre-trained model has been exposed to a vast amount of data, allowing it to capture more general language features, which helps in various language tasks, including ours. Fine-tuning adjusts the model to our specific task, and leaving all parameters trainable, rather than just the last layers, improves performance since every parameter can contribute to minimizing loss.

However, this advantage isn't universal. The success of fine-tuning depends on the alignment between the pre-training data and the target task. For instance, a pre-trained model on image data may not perform well on text tasks compared to a model trained from scratch specifically for text.
"""

part4_q2 = r"""
**Your answer:**

Freezing the last layers of a model while fine-tuning internal layers like multi-headed attention can be less effective than freezing internal layers. This is because the last layers are crucial for adapting the model's output to the new task. If the last layers are not updated, the model might struggle to adjust its task-specific output, leading to poorer performance.
"""


part4_q3= r"""
**Your answer:**

BERT is not suited for machine translation because it lacks a sequence-to-sequence architecture necessary for generating translated output. To use it for this task, you would need to switch to an encoder-decoder model, which handles both encoding the source sequence and decoding the target sequence. Additionally, the model's pre-training would need to focus on tasks that support sequence generation rather than just understanding.
"""

part4_q4 = r"""
**Your answer:**

RNNs might be chosen over Transformers for specific scenarios such as when working with very short sequences or when real-time processing is required. RNNs handle sequences sequentially, which can be advantageous for certain applications despite their potential limitations with long-range dependencies. Transformers, on the other hand, excel in capturing long-range dependencies and parallelizing computations but might be overkill for simpler or shorter tasks where RNNs can be more efficient.
"""

part4_q5 = r"""
**Your answer:**

NSP is a pre-training task used by BERT to learn relationships between sentences. In NSP, the model receives pairs of sentences and predicts whether the second sentence follows the first one in the original text. The prediction is made by a classification head that outputs a binary result: "Is the second sentence the actual next sentence?" or "Is it a random sentence from the corpus?" The loss is computed using binary cross-entropy between the predicted probability and the actual label (1 for a correct pair and 0 for a random pair).

NSP's contribution to pre-training may be limited because it focuses on predicting whether one sentence follows another, which is less relevant for many tasks that don't require sentence order understanding. After some research online, we found that there are more modern training tasks that achieve similar or even better results, without using NSP.
"""


# ==============
