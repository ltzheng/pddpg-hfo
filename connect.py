from envs.hfo_connector import HFOConnector
import os
import click
import argparse
import redis
from utils.redis_manager import connect_redis


# @click.command()
# @click.option('--frames-per-trial', default=1000, type=int)
# @click.option('--start-viewer', default=True, type=bool)
# @click.option('--untouched-time', default=100, type=int)
# @click.option('--offense-agents', default=1, type=int)
# @click.option('--defense-agents', default=0, type=int)
# @click.option('--offense-npcs', default=0, type=int)
# @click.option('--defense-npcs', default=0, type=int)
# @click.option('--sync-mode', default=True, type=bool)
# @click.option('--port', default=None, type=int)
# @click.option('--offense-on-ball', default=0, type=int)
# @click.option('--fullstate', default=True, type=bool)
# @click.option('--seed', default=-1, type=int)
# @click.option('--log-game', default=False, type=bool)
# def configure_environment(frames_per_trial, start_viewer, untouched_time, offense_agents, defense_agents,
#                           offense_npcs, defense_npcs, sync_mode, port, offense_on_ball, fullstate,
#                           seed, log_game):
#     server_args = {'frames_per_trial': frames_per_trial,  # Episodes end after this many steps.
#                    'start_viewer': start_viewer,  # Run with a monitor.
#                    'untouched_time': untouched_time,  # Episodes end if the ball is untouched for this many steps.
#                    'offense_agents': offense_agents,  # Number of user-controlled offensive players.
#                    'defense_agents': defense_agents,  # Number of user-controlled defenders.
#                    'offense_npcs': offense_npcs,  # Number of offensive bots.
#                    'defense_npcs': defense_npcs,  # Number of defense bots.
#                    'sync_mode': sync_mode,  # Disabling sync mode runs server in real time (SLOW!).
#                    'port': port,  # Port to start the server on.
#                    'offense_on_ball': offense_on_ball,  # Player to give the ball to at beginning of episode.
#                    'fullstate': True,  # Enable noise-free perception.
#                    'seed': seed,  # Seed the starting positions of the players and ball.
#                    'ball_x_min': 0.0,  # Initialize the ball this far downfield: [0,1]
#                    'ball_x_max': 0.2,
#                    'verbose': False,  # Verbose server messages.
#                    'log_game': log_game,  # Enable game logging. Logs can be used for replay + visualization.
#                    'log_dir': "log"}  # Directory to place game logs (*.rcg).
#     return server_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offense-agents', type=int, default=1)
    parser.add_argument('--defense-agents', type=int, default=1)
    parser.add_argument('--offense-npcs', type=int, default=0)
    parser.add_argument('--defense-npcs', type=int, default=0)
    parser.add_argument('--server-port', type=int, default=None)
    parser.add_argument('--start-viewer', action='store_true')
    parser.add_argument('--agent-play-goalie', action='store_true')
    parser.add_argument('--no-sync', action='store_false')
    parser.add_argument('--offense-on-ball', type=int, default=0)
    args = parser.parse_args()

    # initialize redis pool
    pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
    r = connect_redis(port=6379)
    r.flushdb()
    r.set('num_agents', args.offense_agents)
    r.set('not ready', args.offense_agents)

    # Create the HFO Environment
    connector = HFOConnector()
    # server_args = configure_environment()
    server_args = {'frames_per_trial': 500,  # Episodes end after this many steps.
                   'start_viewer': args.start_viewer,  # Run with a monitor.
                   'untouched_time': 100,  # Episodes end if the ball is untouched for this many steps.
                   'offense_agents': args.offense_agents,  # Number of user-controlled offensive players.
                   'defense_agents': args.defense_agents,  # Number of user-controlled defenders.
                   'offense_npcs': args.offense_npcs,  # Number of offensive bots.
                   'defense_npcs': args.defense_npcs,  # Number of defense bots.
                   'sync_mode': args.no_sync,  # Disabling sync mode runs server in real time (SLOW!).
                   'port': args.server_port,  # Port to start the server on.
                   'offense_on_ball': args.offense_on_ball,  # Player to give the ball to at beginning of episode.
                   'fullstate': True,  # Enable noise-free perception.
                   'seed': -1,  # Seed the starting positions of the players and ball.
                   'ball_x_min': 0.0,  # Initialize the ball this far downfield: [0,1]
                   'ball_x_max': 0.2,
                   'verbose': False,  # Verbose server messages.
                   'log_game': False,  # Enable game logging. Logs can be used for replay + visualization.
                   'log_dir': "log",
                   'agent_play_goalie': args.agent_play_goalie}  # Directory to place game logs (*.rcg).
    print(server_args)
    connector.start_hfo_server(**server_args)
    # p = subprocess.Popen(['python', os.getcwd() + '/run.py --multipass True --server-port ' + str(connector.server_port)
    #                       + ' --layers [1024,512,256,128] --weighted True --indexed True'], shell=False)
    # p.apply_async(start_agent, args=(configure_agent('offense', port=connector.server_port),))
    # p.apply_async(start_agent, args=(configure_agent('goalie'),))
    # p.close()
    # p.join()
    print('closing...')
