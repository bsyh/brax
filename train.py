import functools
# import jax
from datetime import datetime
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

def train_ant_policy(save_path: str = 'policy/ant_policy_params', lightweight: bool = True) -> callable:
    """
    Train a PPO control policy for the Brax 'ant' environment and save it.

    Args:
        save_path (str): Path to save the trained policy parameters.
        lightweight (bool): If True, use reduced hyperparameters for faster testing.

    Returns:
        callable: Inference function for the trained policy.
    """
    # Environment setup
    env = envs.get_environment(env_name='ant', backend='positional')

    # Training configuration
    if lightweight:
        train_config = functools.partial(
            ppo.train,
            num_timesteps=100_000,
            num_evals=5,
            reward_scaling=10,
            episode_length=100,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=4,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=16,
            batch_size=16,
            seed=1
        )
    else:
        train_config = functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=10,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=4096,
            batch_size=2048,
            seed=1
        )

    # Progress tracking
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Log training progress."""
        print(f"Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

    # Train the policy
    print("Starting training")
    make_inference_fn, params, _ = train_config(environment=env, progress_fn=progress)
    print("Training complete")
    print(f"Time to JIT: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save the trained policy
    model.save_params(save_path, params)
    print(f"Policy saved to {save_path}")

    return make_inference_fn(params)

if __name__ == "__main__":
    # Train and save the policy (lightweight for testing on Apple Silicon)
    inference_fn = train_ant_policy(lightweight=True)