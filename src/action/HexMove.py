class HexMove:

    def __init__(self, row, col, player):
        self.row = row
        self.col = col
        self.player = player
        
    def to_string(self):
        print("Player " + str(self.player) + " filling [row, column]: [" + str(self.row) + ", " + str(self.col) + "]")

# package action;
#
# // Class for the actions done in Ledge
# public class LedgeMove extends AbstractGameAction {
#
#     private final int whatCellToMove;
#     private final int moveLength;
#     private final int player;
#
#     public LedgeMove(int cellToMove, int moveLength, int player) {
#         this.whatCellToMove = cellToMove;
#         this.moveLength = moveLength;
#         this.player = player;
#     }
#
#     public int getCellToMove() {
#         return whatCellToMove;
#     }
#
#     public int getMoveLength() {
#         return moveLength;
#     }
#
#     public int getPlayer() {
#         return player;
#     }
#
#     public String toString() {
#        return "Trying to move CELL: " + whatCellToMove + " " + moveLength + " to the left";
#     }
#
#     public boolean moveRemovesCoinFromLedge() {
#         return getCellToMove() == 0 && getMoveLength() == 1;
#     }
# }
