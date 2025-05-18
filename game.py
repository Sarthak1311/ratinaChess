import pygame
import chess
from GazePrediction import ga
# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8

# Colors
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (0, 255, 0)

# Load piece images
def load_images():
    image_paths = {
        "b_r": "images/blackRook.png",
        "b_n": "images/blackKnight.png",
        "b_b": "images/blackBishop.png",
        "b_q": "images/blackQueen.png",
        "b_k": "images/blackKing.png",
        "b_p": "images/blackPawn.png",
        "w_r": "images/whiteRook.png",
        "w_n": "images/whiteKnight.png",
        "w_b": "images/whiteBishop.png",
        "w_q": "images/whiteQueen.png",
        "w_k": "images/whiteKing.png",
        "w_p": "images/whitePawn.png",
    }
    images = {}
    for key, path in image_paths.items():
        try:
            images[key] = pygame.transform.scale(
                pygame.image.load(path),
                (SQUARE_SIZE, SQUARE_SIZE)
            )
        except:
            print(f"Error loading image: {path}")
    return images

class ChessGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Retina Chess")
        self.board = chess.Board()
        self.images = load_images()
        self.selected_square = None
        self.selected_square_id = None
        self.running = True

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                pygame.draw.rect(
                    self.screen, color,
                    pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )
        
        # Highlight selected square
        if self.selected_square is not None:
            row, col = self.selected_square
            pygame.draw.rect(self.screen, HIGHLIGHT,
                             pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                color = 'w' if piece.color == chess.WHITE else 'b'
                name = f"{color}_{piece.symbol().lower()}"
                image = self.images.get(name)
                if image:
                    self.screen.blit(image, (col*SQUARE_SIZE, row*SQUARE_SIZE))

    def get_square_under_mouse(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        square = chess.square(col, 7 - row)
        return square, (row, col)

    def handle_click(self, pos):
        clicked_square, (row, col) = self.get_square_under_mouse(pos)

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = (row, col)
                self.selected_square_id = clicked_square
        else:
            move = chess.Move(self.selected_square_id, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
            self.selected_square = None
            self.selected_square_id = None

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())

        pygame.quit()

if __name__ == "__main__":
    ChessGame().run()
