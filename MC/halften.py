#!/usr/bin/env python
# encoding: utf-8

import gym
from gym import spaces
from gym.utils import seeding

# Card score(A=1, 2-10=card point, J/Q/K= 0.5)
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5, 0.5, 0.5]

# Person card
p_val = 0.5
# limit value
dest = 10.5


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random)]


def sum_hand(hand):
    return sum(hand)


def get_card_num(hand):
    return len(hand)


def get_p_num(hand):
    count = 0
    for i in hand:
        if i == p_val:
            count += 1
    return count


def gt_bust(hand):
    return sum_hand(hand) > dest


def is_dest(hand):
    return sum_hand(hand) == dest


def lt_dest(hand):
    return sum_hand(hand) < dest


def is_rwx(hand):
    return True if get_p_num(hand) == 5 else False


def is_tw(hand):
    return True if get_card_num(hand) == 5 and is_dest(hand) else False


def is_wx(hand):
    return True if get_card_num(hand) == 5 and lt_dest(hand) else False


def hand_types(hand):
    type = 1
    reward = 0
    done = False

    if gt_bust(hand):
        type = 0
        reward = -1
        done = True
    elif is_rwx(hand):
        type = 5
        reward = 5
        done = True
    elif is_tw(hand):
        type = 4
        reward = 4
        done = True
    elif is_wx(hand):
        type = 3
        reward = 3
        done = True
    elif is_dest(hand):
        type = 2
        reward = 2
        done = True
    return type, reward, done


def cmp(dealer, player):
    dealer_score = sum_hand(dealer)
    player_score = sum_hand(player)

    if dealer_score > player_score:
        return True
    elif dealer_score < player_score:
        return False
    else:
        dealer_num = get_card_num(dealer)
        player_num = get_card_num(player)
        return True if dealer_num >= player_num else False


class HalftenEnv(gym.Env):
    """
    Simple Halften
    Halften is a poker game that is suitable for both young and old.
    The skill of the game lies in how to collect it as 10.5, but if it exceeds ten and a half points, it will fail.
        In the Halften game, the hand (A, 2, 3, 4, 5, 6, 7, 8, 9, 10), A is 1 point, the remaining cards are their own points, the hand (J, Q, K) is a person card, regarded as a half point,
    Now the game in this environment is the dealer and the player.
    Card type description:
    Five small people: 5 cards, each consisting of a card, reward x5
    Uranus: 5 cards, and the total number of cards is 10.5, reward x4
    Five small: 5 cards are not all cards, and the total number of points is less than 10.5, reward x3    10.5: 5 cards or less, the total number of cards is exactly equal to 10.5, reward x2
    Flat card: 5 cards or less, the total number of cards is less than 10.5, reward x1
    Explosion: the total number of cards is greater than 10.5
    Bill rules:
    Card size: People five small> Tianwang> Five small> 10.5> Flat card> Explosion

    If the player gets a card with a card type of 10.5 (or included) (five small, king, five small, ten and a half), he will win immediately, and the dealer will lose.
    If the player gets a card with a total score of more than 10:30, it is a card, the player loses, and the dealer wins immediately.

    If the player gets a card below 10.5 and suspends the card, the dealer will ask for the card and compare it with the player.
    If the current score is smaller than the player, the dealer will continue to ask for the card until the winner is negative. If the dealer equals the player score, the number of the cards is compared. If the number of the cards is less than the number of the player's hand, the player continues to be the card, otherwise the player is determined to win.
    The dealer’s hand also follows the rules of the card.

    Return instructions:
    Winning cards: 1
    Losing cards: -1

    When calculating the return, it should be based on the corresponding magnification of each card type.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(21),
            spaces.Discrete(5),
            spaces.Discrete(6)))
        self._seed()
        self._reset()
        self.nA = 2

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        if action:
            self.player.append(draw_card(self.np_random))

            type, reward, done = hand_types(self.player)
        else:
            done = True
            self.dealer = draw_hand(self.np_random)
            result = cmp(self.dealer, self.player)

            if result:
                reward = -1
            else:
                while not result:
                    self.dealer.append(draw_card(self.np_random))
                    dealer_type, dealer_reward, dealer_done = hand_types(self.dealer)
                    if dealer_done:
                        reward = -dealer_reward
                        break
                    result = cmp(self.dealer, self.player)

                    if result:
                        reward = -1
                        break

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), get_card_num(self.player), get_p_num(self.player))

    def _reset(self):
        self.player = draw_hand(self.np_random)
        return self._get_obs()
