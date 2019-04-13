#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pygame
from pygame.locals import *

# windows font location
FONT_PATH = 'C:/Windows/Fonts/simkai.ttf'


class Board(object):
    '''
    Board game logic control
    '''

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.states = {}  # Board state is a dictionary, key: moving steps, value: player's piece type
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # 5 pieces of a piece win the line
        self.players = [1, 2]  # player 1,2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('The length and width of the board cannot be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        '''
        Returns the position according to the number of moving steps passed (eg: move=2, the calculated coordinates are [0, 2], which means the third horizontal position on the upper left corner of the board)
        :param move:
        :return:
        '''
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        # Return the move value based on the incoming location
        # Location information must contain 2 values [h,w]
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        # Use four 15x15 binary feature planes to describe the current situation
        # The first two planes represent the position of the current player and the position of the opponent's player. The position of the piece is 1, and the position of no piece is 0.
        # The third plane represents the position of the opponent's player in the nearest step, that is, only one position of the entire plane is 1, and the rest are all 0.
        # The fourth plane indicates whether the current player is a first-hand player. If it is a first-hand player, the entire plane is all 1, otherwise all are 0.
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[
                players == self.current_player]  # Get all the movement values belonging to the current player on the board status
            move_oppo = moves[
                players != self.current_player]  # Get all the movement values belonging to the opponent's player on the board status
            square_state[0][move_curr // self.width,  # Fill the value for the first feature plane (current player)
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,  # Fill the value of the second feature plane (the opponent player)
                            move_oppo % self.height] = 1.0
            square_state[2][
                self.last_move // self.width,  # Fill the value of the third feature plane (the last drop position of the opponent)
                self.last_move % self.height] = 1.0
        if len(
                self.states) % 2 == 0:  # For the fourth feature plane fill value, the current player is the first hand, then fill all 1s, otherwise it is all 0
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            # If there are more than 7 chessboards, the winner will be generated. If the number of players is less than 7, the winner will return directly without a winner.
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # 5 horizontal
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 5 vertical
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # 5 left obliquely
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # 5 right obliquely
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


N = 15

IMAGE_PATH = 'UI/'

WIDTH = 540
HEIGHT = 540
MARGIN = 22
GRID = (WIDTH - 2 * MARGIN) / (N - 1)
PIECE = 32


class Game_UI(object):

    def __init__(self, board, is_shown, **kwargs):
        self.board = board
        self.is_shown = is_shown

        pygame.init()

        if is_shown != 0:
            self.__screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
            pygame.display.set_caption('Gomoku AI')

            # UI resource
            self.__ui_chessboard = pygame.image.load(IMAGE_PATH + 'chessboard.jpg').convert()
            self.__ui_piece_black = pygame.image.load(IMAGE_PATH + 'piece_black.png').convert_alpha()
            self.__ui_piece_white = pygame.image.load(IMAGE_PATH + 'piece_white.png').convert_alpha()

    def coordinate_transform_map2pixel(self, i, j):
        '''
        Convert an index to coordinates
        :param i:
        :param j:
        :return:
        '''
        return MARGIN + j * GRID - PIECE / 2, MARGIN + i * GRID - PIECE / 2

    def coordinate_transform_pixel2map(self, x, y):
        '''
        Convert coordinates to index
        :param x:
        :param y:
        :return:
        '''
        i, j = int(round((y - MARGIN + PIECE / 2) / GRID)), int(round((x - MARGIN + PIECE / 2) / GRID))
        if i < 0 or i >= N or j < 0 or j >= N:
            return None, None
        else:
            return i, j

    def draw_chess(self):
        self.__screen.blit(self.__ui_chessboard, (0, 0))
        for i in range(0, N):
            for j in range(0, N):
                # Calculate the movement position
                loc = i * N + j
                p = self.board.states.get(loc, -1)
                player1, player2 = self.board.players

                # Find the coordinates of (i,j)
                x, y = self.coordinate_transform_map2pixel(i, j)

                if p == player1:  # player1 ==> black
                    self.__screen.blit(self.__ui_piece_black, (x, y))
                elif p == player2:  # player2 ==> white
                    self.__screen.blit(self.__ui_piece_white, (x, y))
                else:
                    pass

    def one_step(self):
        i, j = None, None
        mouse_button = pygame.mouse.get_pressed()
        if mouse_button[0]:
            x, y = pygame.mouse.get_pos()
            i, j = self.coordinate_transform_pixel2map(x, y)

        if not i is None and not j is None:
            loc = i * N + j
            p = self.board.states.get(loc, -1)

            player1, player2 = self.board.players

            if p == player1 or p == player2:
                return False
            else:
                cp = self.board.current_player

                location = [i, j]
                move = self.board.location_to_move(location)
                self.board.do_move(move)

                if self.is_shown:
                    if cp == player1:
                        self.__screen.blit(self.__ui_piece_black, (x, y))
                    else:
                        self.__screen.blit(self.__ui_piece_white, (x, y))

                return True
        return False

    def draw_result(self, result):
        font = pygame.font.Font(FONT_PATH, 50)
        tips = u"Game Over:"

        player1, player2 = self.board.players

        if result == player1:
            tips = tips + u"Player 1 wins"
        elif result == player2:
            tips = tips + u"Player 2 wins"
        else:
            tips = tips + u"Tie"
        text = font.render(tips, True, (255, 0, 0))
        self.__screen.blit(text, (WIDTH / 2 - 200, HEIGHT / 2 - 50))

    def start_play_mouse(self, player1, player2, start_player=0):
        if start_player not in (0, 1):
            raise Exception('Must be 0 (player 1) or 1 (player 2)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if start_player != 0:
            current_player = self.board.current_player
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)  # MCTS AI
            self.board.do_move(move)

        if self.is_shown:
            self.draw_chess()
            pygame.display.update()

        flag = False
        win = None

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if self.one_step():
                        end, winner = self.board.game_end()
                    else:
                        continue
                    if end:
                        flag = True
                        win = winner
                        break

                    current_player = self.board.current_player
                    player_in_turn = players[current_player]

                    move = player_in_turn.get_action(self.board)
                    self.board.do_move(move)

                    if self.is_shown:
                        self.draw_chess()
                        pygame.display.update()

                    end, winner = self.board.game_end()
                    if end:
                        flag = True
                        win = winner
                        break

            if flag and self.is_shown:
                self.draw_result(win)
                pygame.display.update()
                break

    def start_play(self, player1, player2, start_player=0):
        if start_player not in (0, 1):
            raise Exception('Must be 0 (player 1) or 1 (player 2)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if self.is_shown:
            self.draw_chess()
            pygame.display.update()

        while True:
            if self.is_shown:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        exit()

            current_player = self.board.current_player
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)  # MCTS AI
            self.board.do_move(move)
            if self.is_shown:
                self.draw_chess()
                pygame.display.update()

            end, winner = self.board.game_end()
            if end:
                win = winner
                break
        if self.is_shown:
            self.draw_result(win)
            pygame.display.update()
        return win

    def start_self_play(self, player, temp=1e-3):
        """
        Use players to start playing games themselves, re-use the search tree and store your own game data
         (state, mcts_probs, z) provide training
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []

        if self.is_shown:
            self.draw_chess()
            pygame.display.update()

        while True:
            if self.is_shown:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if self.is_shown:
                self.draw_chess()
                pygame.display.update()

            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if self.is_shown:
                    self.draw_result(winner)

                    pygame.display.update()
                return winner, zip(states, mcts_probs, winners_z)
