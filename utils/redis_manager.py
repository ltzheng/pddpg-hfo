import redis
import numpy as np


def connect_redis(port=6379):
    return redis.Redis(host='localhost', port=port, decode_responses=True)


# def query_all_obs_actions(redis_instance):
#     all_agents = redis_instance.smembers('teammates')
#     print('all_agents:', all_agents)
#     all_obs = redis_instance.hmget('obs', all_agents)
#     print('all_obs:', all_obs)
#     all_actions = redis_instance.hmget('actions', all_agents)
#     print('all_actions:', all_actions)
#     return all_agents, all_obs, all_actions


# def update_agent(redis_instance, unum, obs, action):
#     redis_instance.hset('obs', 'agent' + str(unum), ' '.join(str(i) for i in obs))
#     redis_instance.hset('actions', 'agent' + str(unum), ' '.join(str(i) for i in action))


# return agent list in string format
def get_all_agents(redis_instance):
    all_agents = list(redis_instance.smembers('teammates'))
    all_agents = [int(i) for i in list(all_agents)]
    all_agents.sort()
    all_agents = [str(i) for i in all_agents]
    # print('all_agents:', all_agents)
    return all_agents


def format_convert(str_list):
    """['1.0 2.0', '3.0 4.0'] -> [[1.0, 2.0], [3.0, 4.0]]"""
    output = []
    if not any(elem is None for elem in str_list):
        for i in str_list:
            output.append([float(j) for j in i.split()])
    else:
        print('Warning: None-type value occurs in redis-server')
        info = False
        return None, info
    output = np.array(output)
    info = True
    return output, info


def query_all_obs_actions(redis_instance):
    all_agents = get_all_agents(redis_instance)

    all_obs, info0 = format_convert(redis_instance.hmget('obs', all_agents))
    # print('all_obs:', all_obs)

    all_agent_act, info1 = format_convert(redis_instance.hmget('actions', all_agents))
    # print('all_agent_actions:', all_agent_actions)

    all_agent_params, info2 = format_convert(redis_instance.hmget('action_params', all_agents))
    # print('all_agent_parameters:', all_agent_params)

    if info1 and info2:
    #     all_agent_actions = []
    #     for all_actions, all_params in zip(all_agent_act, all_agent_params):
    #         all_agent_actions.append(np.concatenate((all_actions.data, all_params.data)))
    #     all_agent_actions = np.array(all_agent_actions)
        all_agent_actions = np.concatenate((all_agent_act.flatten(), all_agent_params.flatten()))
    else:
        all_agent_actions = None

    info = info0 and info1 and info2
    return all_obs, all_agent_actions, info


def sync_agent_obs_actions(redis_instance, unum, obs, all_actions, all_action_parameters):
    redis_instance.hset('obs', str(unum), ' '.join(str(i) for i in obs))
    redis_instance.hset('actions', str(unum), ' '.join(str(i) for i in all_actions))
    redis_instance.hset('action_params', str(unum), ' '.join(str(i) for i in all_action_parameters))


# query other agents' policies when not inferring them
def sync_agent_policy(redis_instance, unum, actor_network):
    redis_instance.hset('policy', str(unum), ' '.join(str(i) for i in actor_network.parameters()))


def query_all_policies(redis_instance):
    all_agents = get_all_agents(redis_instance)
    all_agent_policy = format_convert(redis_instance.hmget('policy', all_agents))
    return all_agent_policy


# def query_agent_obs(redis_instance, agent):
#     obs, info = format_convert(redis_instance.hmget('obs', agent))
#     return obs, info


def get_agent_idx(redis_instance, agent_unum):
    all_agents = get_all_agents(redis_instance)
    return all_agents.index(agent_unum)


# # TODO: query other agents' actions (for inference use)
# other_agents = self.redis_instance.smembers('teammates')
# print('other_agents:', other_agents)  # works well
# other_agents = other_agents.remove(self.env.unum)  # where problem occurs
# print('other_agents:', other_agents)
# other_actions = self.redis_instance.hmget('actions', other_agents)