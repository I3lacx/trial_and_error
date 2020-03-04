"""
Main file for running the whole game, calling engine
and network
"""
import game_engine


def main():
    game = game_engine.Game()
    while game.running:
        game.run()
    print("Finished main loop")


if __name__ == "__main__":
    main()
