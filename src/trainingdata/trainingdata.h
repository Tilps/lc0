/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include "mcts/node.h"
#include "trainingdata/writer.h"

namespace lczero {

#pragma pack(push, 1)

struct V6TrainingData {
  uint32_t version;
  uint32_t input_format;
  uint64_t planes[110];
  float result_q;
  float result_d;
  uint32_t p1idx;
  uint32_t p2idx;
} PACKED_STRUCT;
static_assert(sizeof(V6TrainingData) == 888+16, "Wrong struct size");

#pragma pack(pop)

class V6TrainingDataArray {
 public:
  V6TrainingDataArray(FillEmptyHistory white_fill_empty_history,
                      FillEmptyHistory black_fill_empty_history,
                      pblczero::NetworkFormat::InputFormat input_format)
      : fill_empty_history_{white_fill_empty_history, black_fill_empty_history},
        input_format_(input_format) {}

  // Add a chunk.
  void Add(const Node* node, const PositionHistory& history, Eval best_eval,
           Eval played_eval, bool best_is_proven, Move best_move,
           Move played_move, const NNCacheLock& nneval);

  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;

 private:
  std::vector<V6TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
};

}  // namespace lczero
