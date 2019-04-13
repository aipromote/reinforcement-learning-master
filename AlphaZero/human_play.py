#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from game import Board, Game_UI
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet  # Tensorflow


class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p


def run():
    n = 5
    width, height = 15, 15
    model_file = 'dist/best_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game_UI(board, is_shown=1)

        # ############### Human-machine ###################
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        human = Human()

        game.start_play_mouse(human, mcts_player, start_player=0, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
