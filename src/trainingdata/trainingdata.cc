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

#include "trainingdata/trainingdata.h"

namespace lczero {

namespace {
void DriftCorrect(float* q, float* d) {
  // Training data doesn't have a high number of nodes, so there shouldn't be
  // too much drift. Highest known value not caused by backend bug was 1.5e-7.
  const float allowed_eps = 0.000001f;
  if (*q > 1.0f) {
    if (*q > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in q " << *q;
    }
    *q = 1.0f;
  }
  if (*q < -1.0f) {
    if (*q < -1.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in q " << *q;
    }
    *q = -1.0f;
  }
  if (*d > 1.0f) {
    if (*d > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in d " << *d;
    }
    *d = 1.0f;
  }
  if (*d < 0.0f) {
    if (*d < 0.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in d " << *d;
    }
    *d = 0.0f;
  }
  float w = (1.0f - *d + *q) / 2.0f;
  float l = w - *q;
  // Assume q drift is rarer than d drift and apply all correction to d.
  if (w < 0.0f || l < 0.0f) {
    float drift = 2.0f * std::min(w, l);
    if (drift < -allowed_eps) {
      CERR << "Unexpectedly large drift correction for d based on q. " << drift;
    }
    *d += drift;
    // Since q is in range -1 to 1 - this correction should never push d outside
    // of range, but precision could be lost in calculations so just in case.
    if (*d < 0.0f) {
      *d = 0.0f;
    }
  }
}
}  // namespace


void V6TrainingDataArray::Write(TrainingDataWriter* writer, GameResult result,
                                bool adjudicated) const {
  if (training_data_.empty()) return;
}

void V6TrainingDataArray::Add(const Node* node, const PositionHistory& history,
                              Eval best_eval, Eval played_eval,
                              bool best_is_proven, Move best_move,
                              Move played_move, const NNCacheLock& nneval) {

}

}  // namespace lczero
