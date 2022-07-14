def get_card_from_id(card_id):
    card_types = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    card_symbols = ['Club', 'Diamond', 'Heart', 'Spade'] # club, diamond, heart, spade = ['♣', '♦', '♥', '♠']
    card_names = gen_cardnames(card_types, card_symbols)  # card class array (defined by card names)
    return card_names[card_id]


def gen_cardnames(card_types, card_symbols):
    inumsymb = len(card_symbols)

    inumtypes = len(card_types)
    card_names = []
    for ii in range(inumsymb):
        for jj in range(inumtypes):
            card_names.append(card_types[jj] + " of " + card_symbols[ii])
    return card_names
