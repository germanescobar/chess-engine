import * as fs from 'fs'
import { Chess } from 'chess.js'
import * as tf from '@tensorflow/tfjs-node-gpu'

let trained = false
run().then(() => console.log("finished!"))
  
async function run() {
  const model = await loadModel()
  for (let i=0; i < 40; i++) {
    const games = playGames(model, 50)
    const data = []
    const results = []
    for (let i=0; i < games.length; i++) {
      const moves = games[i]
      for (let j=0; j < moves.length; j++){
        const move = moves[j]
        data.push(encodeBoard(move.position, move.move.color))
        const result = encodeMove(move.move)
        result.push(move.winningProbability)
        results.push(result)
      }
    }
    console.log("Data length: ", data[0].length)
    console.log("Result length:", results[0].length)
  
    await train(model, data, results)
    trained = true
    if (i % 20 === 0) await model.save("file://model")
  }

  await model.save("file://model")
}

async function loadModel() {
  const ALPHA = 0.001

  let model
  if (fs.existsSync('model/model.json')) {
    console.log("Loading from existing model ...")
    model = await tf.loadLayersModel('file://model/model.json')
    trained = true
  } else {
    model = tf.sequential({
      layers: [
        tf.layers.dense({inputShape: [770], units: 2048, activation: 'relu'}),
        tf.layers.dense({units: 4097, activation: 'softmax'}),
      ]
    });
  }

  model.compile({
    optimizer: tf.train.sgd(ALPHA),
    loss: "meanSquaredError",
  })

  return model
}

function playGames(model, numGames) {
  const games = []
  for (let i=0; i < numGames; i++) {
    const chess = new Chess()
    const moves = []
    let movesWithoutCaptures = 0
    while (!chess.isGameOver() && movesWithoutCaptures < 50) {
      const position = getBoard(chess)
      const m = getMove(model, chess)
      const move = chess.move(m)
      moves.push({ position, move })
      if (move.flags.indexOf("c") === -1) {
        movesWithoutCaptures++
      } else {
        movesWithoutCaptures = 0
      }
    }
    
    console.log("Moves: ", moves.length)
    const lastMove = moves[moves.length - 1]
    const winningColor = lastMove.move.color
    let prob = 1
    const discount = 0.55 / moves.length / 2
    for (let i=moves.length-1; i >= 0; i--) {
      const move = moves[i]
      if (chess.isCheckmate()) {
        move.winningProbability = move.color === winningColor ? prob : 1 - prob
        prob -= discount
      } else {
        move.winningProbability = 0.5
      }
    }
    games.push(moves)
  }
  return games
}

async function train(model, data, results) {
  await model.fit(tf.tensor(data), tf.tensor(results))

  console.log(model.layers[0].getWeights()[0].shape)
  model.layers[0].getWeights()[0].print()
  console.log(model.layers[1].getWeights()[0].shape)
  model.layers[1].getWeights()[0].print()
}

function getMove(model, chess) {
  if (trained) {
    const validMoves = chess.moves({ verbose: true })
    const input = encodeBoard(getBoard(chess), validMoves[0].color)
    const output = model.predict(tf.tensor([input]))
    const result = output.dataSync()
    
    let sum = 0
    for (let i=0; i < validMoves.length; i++) {
      const move = validMoves[i]
      const encodedFrom = encodeCell(move.from)
      const encodedTo = encodeCell(move.to)

      move.value = result[encodedFrom * 64 + encodedTo]

      if (!move.value) {
        console.log("No value for move:", move)
      } else {
        sum += move.value
      }
    }

    const weights = []
    for (let i=0; i < validMoves.length; i++) {
      const move = validMoves[i]
      weights.push(move.value / sum)
    }
    // console.log("Weights:", weights)
    const selectedMove = weightedRandom(validMoves, weights)
    
    if (!selectedMove) {
      console.log("No selected move")
      return validMoves[0]
    }
    // console.log(selectedMove)
    return selectedMove
  } else {
    const moves = chess.moves({ verbose: true })
    return moves[Math.floor(Math.random() * moves.length)]
  }
}

function weightedRandom(items, weights) {
  let i;
  for (i = 0; i < weights.length; i++) {
    weights[i] += weights[i - 1] || 0;
  }
  const random = Math.random();
  for (i = 0; i < weights.length; i++) {
    if (weights[i] > random) break;
  }
      
  return items[i];
}

function getBoard(chess) {
  const board = chess.board()
  let result = []
  for (let i=0; i < board.length; i++) {
    const row = []
    for (let j=0; j < board[i].length; j++) {
      const obj = board[i][j]
      row.push(obj ? `${obj.color}${obj.type}` : null)
    }
    result.push(row)
  }
  return result
}

function encodeBoard(board, color) {
  const mapping = {
    wk: 0, // white king
    bk: 1, // black king
    wq: 2, // white queen
    bq: 3, // black queen
    wr: 4, // white rook
    br: 5, // black rook
    wb: 6, // white bishop
    bb: 7, // black bishop
    wn: 8, // white knight
    bn: 9, // black knight
    wp: 10, // white pawn
    bp: 11 // black pawn
  }
   
  let result = []
  for (let i=0; i < board.length; i++) {
    for (let j=0; j < board[i].length; j++) {
      const type = board[i][j]
      const cell = Array(12).fill(0)
      if (type) {
        const idx = mapping[type]
        cell[idx] = 1
      }
      result = result.concat(cell)
    }
  }
 
  result = result.concat(color === "w" ? [1, 0] : [0, 1] )
  return result
}

function encodeMove(move) {
  const from = encodeCell(move.from)
  const to = encodeCell(move.to)

  const mat = []
  for (let i=0; i < 64; i++) {
    const row = Array(64).fill(0)
    mat.push(row)
  }
  mat[from][to] = 1

  const result = []
  for (let i=0; i < mat.length; i++) {
    for (let j=0; j < mat[i].length; j++) {
      result.push(mat[i][j])
    }
  }
  return result
}

function encodeCell(cell) {
  const col = cell.charCodeAt(0) - 97
  const row = parseInt(cell[1]) - 1
  return col + (row * 8)
}