from gesticulator.model.diffusion.respace import SpacedDiffusion, space_timesteps
from model.model import GesticulatorModel
import model.diffusion.gaussian_diffusion as gd

# def create_diffusion_model(args, data):
#     ## TODO: CHANGE THE GET_MODEL_ARGS
#     model = GesticulatorModel(**get_model_args(args, data))
#     ## Prepare all the parameters and functions needed for the diffusion process.
#     diffusion = create_gaussian_diffusion(args)
#     return model, diffusion

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling ## Beta controls the rate of diffusion (how much noise do we add)
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it. ## Determines how often does the diffusion step take place.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta) ## Choose the correspondent beta-scheduler
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ## Depending on where we are, the model predicts the added noise or all the original noise.
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ## Set the variance of the models
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )

