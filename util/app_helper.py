card_types = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
card_symbols = ['clubs', 'diamonds', 'hearts', 'spades']  # club, diamond, heart, spade = ['♣', '♦', '♥', '♠']


def get_card_from_id(card_id):
    card_names = gen_card_names()  # card class array (defined by card names)
    return card_names[card_id]


def gen_card_names():
    inumsymb = len(card_symbols)

    inumtypes = len(card_types)
    card_names = []
    for ii in range(inumsymb):
        for jj in range(inumtypes):
            card_names.append(card_types[jj] + "_of_" + card_symbols[ii])
    return card_names


def get_card_value(card_id):
    if card_id % len(card_types) + 2 > 10:
        return 10
    return card_id % len(card_types) + 2
