import cv2
import util.app_helper as ah
import service.image_loader as il
import numpy as np
import random
import util.constants as constants

player_score = 0
pc_score = 0
generated_cards = []


def compute_score(card_class):
    global player_score
    player_score += ah.get_card_value(card_class)


def show_score(img):
    cv2.putText(img,
                str("Player Score: %d" % player_score),
                (img.shape[0]-15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(img,
                str("PC Score: %d" % pc_score),
                (img.shape[0]-15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    if len(generated_cards) > 1:
        if (21 >= player_score > pc_score) or (pc_score > 21 and player_score <= 21):
            cv2.putText(img,
                        str("Player WON :)"),
                        (img.shape[0]-15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(img,
                        str("PC won :("),
                        (img.shape[0]-15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


def reset_score():
    global player_score, pc_score
    player_score = 0
    pc_score = 0


def play_card(img, cards_in_image):
    global pc_score
    pc_score = 0
    cards_to_generate = len(cards_in_image)
    generated_cards_number = len(generated_cards)

    for card_pos in range(0, generated_cards_number):
        # if card_pos < cards_to_generate:
        draw_card(img, generated_cards[card_pos], card_pos)

    if (generated_cards_number >= cards_to_generate and player_score <= pc_score) or pc_score > 18 or player_score > 21:
        return
    else:
        card_id = generate_random_card()
        draw_card(img, card_id, generated_cards_number)


def draw_card(img, card_id, pos):
    global pc_score
    card_size = (100, 160)
    card_name = ah.get_card_from_id(card_id)
    pc_score += ah.get_card_value(card_id)

    path = "../resources/images/ssip_20k_cards/assets/" + card_name + ".png"
    card = cv2.resize(il.load_image_matplot(path)[..., :3] * 255, card_size,
                      interpolation=cv2.INTER_AREA)
    # bottom left corner
    (x, y) = (10, 10)
    # (x, y) = (img.shape[0], 0)
    (height, width, _) = card.shape
    if pos > 0:
        img[x:x + height, y + width*pos+(10*pos):y + width*(pos+1)+(10*pos)] = card
    else:
        img[x:x + height, y:y + width] = card


def generate_random_card():
    random_class = random.randrange(0, constants.number_of_classes-1)

    while random_class in generated_cards:
        random_class = random.randrange(0, constants.number_of_classes-1)

    generated_cards.append(random_class)
    return random_class


def reset_generated_cards():
    global generated_cards
    generated_cards = []