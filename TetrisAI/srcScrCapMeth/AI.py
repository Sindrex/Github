from srcScrCapMeth.Interaction import *

def gen_data_prologue(states=[], actions=[], metrics=[]):
    taking_turns = True
    while taking_turns:
        state = []
        act = []

        # Grab Board State
        select_next_unit()
        name = grab_name()
        act.append(name)
        board_state = grab_board_state()

        metric = board_state[1] + board_state[2]

        state.append(board_state[0])
        state.append(board_state[1])
        state.append(board_state[2])

        # Set Options on 1st Turn Only
        if state[2] == 1:
            set_options()
            time.sleep(0.2)

        # Execute Actions
        select_next_unit()
        d = grab_stats()
        df = pd.DataFrame(d, index=[name], columns=d.keys())
        state.extend(df.values[0])

        moves = move_unit(random_move=True)
        act.extend(list(moves))
        opts = grab_options()
        if not opts:
            act.append("Invalid move")
            print('Broken for Invalid Move')
            break
        else:
            selection = choose_option(opts, random_opt=True)
            act.append(selection)

        metrics.append(metric)
        states.append(state)
        actions.append(act)

        # Break loop if turn count greater than 8
        if state[2] > 8:
            print('Broken for exceeding turn count')
            break
    return states, actions, metrics