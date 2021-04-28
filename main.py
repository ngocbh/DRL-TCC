import random
import argparse
import os
import torch
import time
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MCActor, Critic
from environment import WRSNEnv
from vec_env import make_vec_envs
from utils import NetworkInput, WRSNDataset, Point
from utils import Config, DrlParameters as dp, WrsnParameters as wp
from utils import logger, gen_cgrg, device, writer, device_str


def validate(data_loader, actor, render=False, verbose=False):
    actor.eval()

    rewards = []
    mean_policy_losses = []
    mean_entropies = []
    times = [0]
    net_lifetimes = []
    mc_travel_dists = []
    mean_aggregated_ecrs = []
    mean_node_failures = []

    for idx, data in enumerate(data_loader):
        if verbose: print("Test %d" % idx)

        sensors, targets = data
        
        env = WRSNEnv(sensors=sensors.squeeze(), 
                      targets=targets.squeeze(), 
                      normalize=True)

        mc_state, depot_state, sn_state = env.reset()
        mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
        depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
        sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

        rewards = []
        aggregated_ecrs = []
        node_failures = []

        mask = torch.ones(env.action_space.n).to(device)

        for step in range(dp.max_step):
            if render:
                env.render()

            mc_state = mc_state.unsqueeze(0)
            depot_state = depot_state.unsqueeze(0)
            sn_state = sn_state.unsqueeze(0)

            with torch.no_grad():
                logit = actor(mc_state, depot_state, sn_state)

            logit = logit + mask.log()
            prob = F.softmax(logit, dim=-1)

            prob, action = torch.max(prob, 1)  # Greedy selection
            

            mask[env.last_action] = 1.0
            (mc_state, depot_state, sn_state), reward, done, _ = env.step(action.squeeze().item())
            mask[env.last_action] = 0.0

            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            if verbose: 
                print("Step %d: Go to %d (prob: %2.4f) => reward (%2.4f, %2.4f)\n" % 
                      (step, action, prob, reward[0], reward[1]))
                print("Current network lifetime: %2.4f \n\n" % env.net.network_lifetime)

            rewards.append(reward)
            aggregated_ecrs.append(env.net.aggregated_ecr)
            node_failures.append(env.net.node_failures)

            if done:
                if verbose: print("End episode! Press any button to continue...")
                if render:
                    env.render()
                    input()
                env.close()
                break

            if render:
                time.sleep(0.5)
                # pass

        net_lifetimes.append(env.get_network_lifetime())
        mc_travel_dists.append(env.get_travel_distance())
        mean_aggregated_ecrs.append(np.mean(aggregated_ecrs))
        mean_node_failures.append(np.mean(node_failures))

    ret = {}
    ret['lifetime_mean'] = np.mean(net_lifetimes)
    ret['lifetime_std'] = np.std(net_lifetimes)
    ret['travel_dist_mean'] = np.mean(mc_travel_dists)
    ret['travel_dist_std'] = np.std(mc_travel_dists)
    ret['aggregated_ecr_mean'] = np.mean(mean_aggregated_ecrs)
    ret['aggregated_ecr_std'] = np.std(mean_aggregated_ecrs)
    ret['node_failures_mean'] = np.mean(mean_node_failures)
    ret['node_failures_std'] = np.std(mean_node_failures)

    return ret


def train(actor, critic, train_data, valid_data, save_dir, epoch_start_idx=0):
    logger.info("Begin training phase")
    train_loader = DataLoader(train_data, dp.batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, 1, False, num_workers=0)

    actor_optim = optim.Adam(actor.parameters(), dp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), dp.critic_lr)

    best_params = None
    best_reward = np.inf
    sample_inp = None

    for epoch in range(epoch_start_idx, dp.num_epoch):
        logger.info("Start epoch %d" % epoch)
        actor.train()
        critic.train()

        epoch_start = time.time()
        start = epoch_start

        mean_policy_losses = []
        mean_entropies = []
        times = [0]
        net_lifetimes = []
        mc_travel_dists = []

        for idx, data in enumerate(train_loader):
            sensors, targets = data
            batch_size, sequence_size, _ = sensors.size()

            envs = make_vec_envs(sensors, targets, normalize=True)

            mc_state, depot_state, sn_state = envs.reset()
            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            if sample_inp is None:
                sample_inp = (mc_state, depot_state, sn_state)

            values = []
            log_probs = []
            rewards = []
            entropies = []
            dones = []

            mask = torch.ones(batch_size, envs.action_space.n).to(device)

            for _ in range(dp.max_step):
                logit = actor(mc_state, depot_state, sn_state)
                logit = logit + mask.log()

                prob = F.softmax(logit, dim=-1)

                value = critic(mc_state, depot_state, sn_state)

                m = torch.distributions.Categorical(prob)
                
                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                action = m.sample()
                logp = m.log_prob(action)
                entropy = m.entropy()
                last_action = envs.get_attr('last_action')
                mask[range(batch_size), last_action] = torch.ones(batch_size)
                
                envs.step_async(action.detach().numpy())
                (mc_state, depot_state, sn_state), reward, done, info = envs.step_wait()

                last_action = envs.get_attr('last_action')
                mask[range(batch_size), last_action] = torch.zeros(batch_size)

                mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
                depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
                sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

                values.append(value)
                rewards.append(reward)
                log_probs.append(logp)
                entropies.append(entropy)
                dones.append(done)
            
            net_lifetime = envs.env_method('get_network_lifetime')
            net_lifetimes.extend(net_lifetime)

            mc_travel_dist = envs.env_method('get_travel_distance')
            mc_travel_dists.extend(mc_travel_dist)
            envs.close()

            R = torch.zeros(batch_size, 1).to(device)

            value = critic(mc_state, depot_state, sn_state)

            values.append(value)
            
            gae = torch.zeros(batch_size, 1).to(device)
            policy_losses = torch.zeros(len(rewards), batch_size, 1)
            value_losses = torch.zeros(len(rewards), batch_size, 1)

            R = values[-1]

            for i in reversed(range(len(rewards))):
                values[i+1][dones[i]] = 0.0

                reward = rewards[i][:, 0].reshape(-1, 1) # using lifetime only
                reward = torch.tensor(reward)
                R = dp.gamma * R + reward
                
                advantage = R - values[i]
                value_losses[i] = 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = reward + dp.gamma * \
                    values[i + 1] - values[i]

                gae = gae * dp.gamma * dp.gae_lambda + delta_t
                policy_losses[i] = -log_probs[i].view(-1, 1) * gae.detach() - \
                                     dp.entropy_coef * entropies[i].view(-1, 1)
                

            actor_optim.zero_grad()
            policy_losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), dp.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            value_losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), dp.max_grad_norm)
            critic_optim.step()

            with torch.no_grad():
                pl = torch.mean(policy_losses).item()
                mean_policy_losses.append(pl)
                vl = torch.mean(value_losses).item()
                e = torch.mean(torch.stack(entropies)).item()                
                mean_entropies.append(e)


            if (idx + 1) % dp.log_size == 0:
                end = time.time()
                times.append(end-start)
                start = end

                mm_policy_loss = np.mean(mean_policy_losses[-100:])
                mm_entropies = np.mean(mean_entropies[-100:])
                m_net_lifetime = np.mean(net_lifetimes[-100:])
                m_mc_travel_dist = np.mean(mc_travel_dists[-100:])
                global_step = idx/100 + epoch * len(train_loader)
                writer.add_scalar('batch/policy_loss', mm_policy_loss, global_step)
                writer.add_scalar('batch/entropy', mm_entropies, global_step)
                writer.add_scalar('batch/net_lifetime', m_net_lifetime, global_step)
                writer.add_scalar('batch/mc_travel_dist', m_mc_travel_dist, global_step)

                msg = '\tBatch %d/%d, mean_policy_losses: %2.3f, ' + \
                    'mean_net_lifetime: %2.4f, mean_mc_travel_dist: %2.4f, ' + \
                    'mean_entropies: %2.4f, took: %2.4fs'
                logger.info(msg % (idx, len(train_loader), mm_policy_loss, 
                                   m_net_lifetime, m_mc_travel_dist,
                                   mm_entropies, times[-1]))

        mm_policy_loss = np.mean(mean_policy_losses)
        mm_entropies = np.mean(mean_entropies)
        m_net_lifetime = np.mean(net_lifetimes)
        m_mc_travel_dist = np.mean(mc_travel_dists)


        # Save the weights
        epoch_dir = os.path.join(save_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        res = validate(valid_loader, actor)
        m_net_lifetime_valid = res['lifetime_mean'] 
        m_mc_travel_dist_valid = res['travel_dist_mean']

        writer.add_scalar('epoch/policy_loss', mm_policy_loss, epoch)
        writer.add_scalar('epoch/entropy', e, epoch)
        writer.add_scalars('epoch/net_lifetime', 
                           {'train': m_net_lifetime,
                            'valid': m_net_lifetime_valid},
                           epoch)
        writer.add_scalars('epoch/mc_travel_dist', 
                           {'train': m_mc_travel_dist,
                            'valid': m_mc_travel_dist_valid},
                           epoch)

        # Save best model parameters
        if m_net_lifetime_valid < best_reward:

            best_reward = m_net_lifetime_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        msg = 'Epoch %d: mean_policy_losses: %2.3f, ' + \
            'mean_net_lifetime: %2.4f, mean_mc_travel_dist: %2.4f, ' + \
            'mean_entropies: %2.4f, m_net_lifetime_valid: %2.4f, ' + \
            'took: %2.4fs, (%2.4f / 100 batches)\n'
        logger.info(msg % (epoch, mm_policy_loss, m_net_lifetime, 
                           m_mc_travel_dist, mm_entropies, m_net_lifetime_valid,
                           time.time() - epoch_start, np.mean(times)))

    writer.add_graph(actor, sample_inp)


def main(num_sensors=20, num_targets=10, config=None,
         checkpoint=None, save_dir='checkpoints', seed=123, 
         mode='train', epoch_start=0, render=False, verbose=False):
    logger.info("Running problem with %d sensors %d targets: " + 
                "(checkpoint: %s, seed : %d, config: %s)", 
                num_sensors, num_targets, checkpoint, seed, config or 'default')

    if config is not None:
        wp.from_file(config)
        dp.from_file(config)

    if config is not None:
        basefile = os.path.splitext(os.path.basename(config))[0]
    else:
        basefile = 'default'
    save_dir = os.path.join(save_dir, basefile)

    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE, 
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    critic = Critic(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size).to(device)

    if checkpoint is not None:
        path = os.path.join(checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if mode == 'train':
        logger.info("Generating training dataset")
        train_data = WRSNDataset(num_sensors, num_targets, dp.train_size, seed)
        logger.info("Generating validation dataset")
        valid_data = WRSNDataset(num_sensors, num_targets, dp.valid_size, seed + 1)
        train(actor, critic, train_data, valid_data, save_dir, epoch_start)

    test_data = WRSNDataset(num_sensors, num_targets, dp.test_size, seed)
    test_loader = DataLoader(test_data, 1, False, num_workers=0)

    ret = validate(test_loader, actor, render, verbose)
    lifetime, travel_dist = ret['lifetime_mean'], ret['travel_dist_mean']

    logger.info("Test metrics: Mean network lifetime %2.4f, mean travel distance: %2.4f",
                lifetime, travel_dist)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Mobile Charger Trainer")
    parser.add_argument('--num_sensors', '-ns', default=20, type=int)
    parser.add_argument('--num_targets', '-nt', default=10, type=int)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'eval'])
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--checkpoint', '-cp', default=None, type=str)
    parser.add_argument('--save_dir', '-sd', default='checkpoints', type=str)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    
    logger.info("Running on device: %s", device_str)
    torch.set_printoptions(sci_mode=False)
    seed = 46
    torch.manual_seed(seed)
    np.set_printoptions(suppress=True)

    main(**vars(args))
