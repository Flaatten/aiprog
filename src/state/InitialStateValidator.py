class InitialStateValidator:

    @staticmethod
    def is_valid(state):
        # TODO
        pass


# package state;
#
# public class InitialStateValidator {
#
#     public static boolean isValid(TwoPlayerAbstractGameState state) {
#         if (state instanceof LedgeGameState) {
#             int numGoldCoins = 0;
#
#             // Contains a single gold coin
#             for (Integer val : ((LedgeGameState) state).getBoard()) {
#                 if (val == LedgeGameState.GOLD_COIN) {
#                     numGoldCoins++;
#                 }
#             }
#
#             // Values within 0-2 (Open, Cobber, Gold)
#             for (Integer val : ((LedgeGameState) state).getBoard()) {
#                 if (val < 0 || val > 2) {
#                     return false;
#                 }
#             }
#
#             return numGoldCoins == 1;
#         } else if (state instanceof NIMGameState) {
#             return ((NIMGameState) state).getNumPiecesLeft() > 0;
#         } else {
#             throw new IllegalArgumentException("State object not defind within InitialStateValidator");
#         }
#     }
#
# }
