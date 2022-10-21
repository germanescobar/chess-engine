import { Chess } from 'chess.js'

const chess = new Chess()

const moves = chess.moves()
// console.log(moves)

chess.move("sdsd")

console.log(chess.fen())