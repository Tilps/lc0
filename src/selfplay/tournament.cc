/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "selfplay/tournament.h"

#include "chess/pgn.h"
#include "mcts/search.h"
#include "mcts/stoppers/factory.h"
#include "neural/factory.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{
    "games", "Games",
    "Number of games to play. -1 to play forever, -2 to play equal to book "
    "length, or double book length if mirrored."};
const OptionId kParallelGamesId{"parallelism", "Parallelism",
                                "Number of games to play in parallel."};
const OptionId kThreadsId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use for every game,", 't'};
const OptionId kPlayoutsId{"playouts", "Playouts",
                           "Number of playouts per move to search."};
const OptionId kVisitsId{"visits", "Visits",
                         "Number of visits per move to search."};
const OptionId kTimeMsId{"movetime", "MoveTime",
                         "Time per move, in milliseconds."};
const OptionId kTrainingId{
    "training", "Training",
    "Enables writing training data. The training data is stored into a "
    "temporary subdirectory that the engine creates."};
const OptionId kVerboseThinkingId{"verbose-thinking", "VerboseThinking",
                                  "Show verbose thinking messages."};
const OptionId kMoveThinkingId{"move-thinking", "MoveThinking",
                               "Show all the per-move thinking."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};
const OptionId kDiscardedStartChanceId{
    "discarded-start-chance", "DiscardedStartChance",
    "The percentage chance each game will attempt to start from a position "
    "discarded due to not getting enough visits."};
const OptionId kOpeningsFileId{
    "openings-pgn", "OpeningsPgnFile",
    "A path name to a pgn file containing openings to use."};
const OptionId kOpeningsMirroredId{
    "mirror-openings", "MirrorOpenings",
    "If true, each opening will be played in pairs. "
    "Not really compatible with openings mode random."};
const OptionId kOpeningsModeId{"openings-mode", "OpeningsMode",
                               "A choice of sequential, shuffled, or random."};
}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");
  options->AddContext("white");
  options->AddContext("black");
  for (const auto context : {"player1", "player2"}) {
    auto* dict = options->GetMutableOptions(context);
    dict->AddSubdict("white")->AddAliasDict(&options->GetOptionsDict("white"));
    dict->AddSubdict("black")->AddAliasDict(&options->GetOptionsDict("black"));
  }

  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(options);

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -2, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<BoolOption>(kMoveThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kDiscardedStartChanceId, 0.0f, 100.0f) = 0.0f;
  options->Add<StringOption>(kOpeningsFileId) = "";
  options->Add<BoolOption>(kOpeningsMirroredId) = false;
  std::vector<std::string> openings_modes = {"sequential", "shuffled",
                                             "random"};
  options->Add<ChoiceOption>(kOpeningsModeId, openings_modes) = "sequential";

  SelfPlayGame::PopulateUciParams(options);

  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(SearchParams::kMiniBatchSizeId, 32);
  defaults->Set<float>(SearchParams::kCpuctId, 1.2f);
  defaults->Set<float>(SearchParams::kCpuctFactorId, 0.0f);
  defaults->Set<float>(SearchParams::kPolicySoftmaxTempId, 1.0f);
  defaults->Set<int>(SearchParams::kMaxCollisionVisitsId, 1);
  defaults->Set<int>(SearchParams::kMaxCollisionEventsId, 1);
  defaults->Set<int>(SearchParams::kCacheHistoryLengthId, 7);
  defaults->Set<bool>(SearchParams::kOutOfOrderEvalId, false);
  defaults->Set<float>(SearchParams::kTemperatureId, 1.0f);
  defaults->Set<float>(SearchParams::kNoiseEpsilonId, 0.25f);
  defaults->Set<float>(SearchParams::kFpuValueId, 0.0f);
  defaults->Set<std::string>(SearchParams::kHistoryFillId, "no");
  defaults->Set<std::string>(NetworkFactory::kBackendId, "multiplexing");
  defaults->Set<bool>(SearchParams::kStickyEndgamesId, false);
}

SelfPlayTournament::SelfPlayTournament(
    const OptionsDict& options,
    CallbackUciResponder::BestMoveCallback best_move_info,
    CallbackUciResponder::ThinkingCallback thinking_info,
    GameInfo::Callback game_info, TournamentInfo::Callback tournament_info)
    : player_options_{{options.GetSubdict("player1").GetSubdict("white"),
                       options.GetSubdict("player1").GetSubdict("black")},
                      {options.GetSubdict("player2").GetSubdict("white"),
                       options.GetSubdict("player2").GetSubdict("black")}},
      best_move_callback_(best_move_info),
      info_callback_(thinking_info),
      game_callback_(game_info),
      tournament_callback_(tournament_info),
      kTotalGames(options.Get<int>(kTotalGamesId)),
      kShareTree(options.Get<bool>(kShareTreesId)),
      kParallelism(options.Get<int>(kParallelGamesId)),
      kTraining(options.Get<bool>(kTrainingId)),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId)),
      kDiscardedStartChance(options.Get<float>(kDiscardedStartChanceId)) {
  std::string book = options.Get<std::string>(kOpeningsFileId);
  if (!book.empty()) {
    PgnReader book_reader;
    book_reader.AddPgnFile(book);
    openings_ = book_reader.ReleaseGames();
  }
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    first_game_black_ = Random::Get().GetBool();
  }

  // Initializing networks.
  for (const auto& name : {"player1", "player2"}) {
    for (const auto& color : {"white", "black"}) {
      const auto& opts = options.GetSubdict(name).GetSubdict(color);
      const auto config = NetworkFactory::BackendConfiguration(opts);
      if (networks_.find(config) == networks_.end()) {
        networks_.emplace(config, NetworkFactory::LoadNetwork(opts));
      }
    }
  }
  for (int i = 0; i < openings_.size(); i += 320) {
    auto comp = networks_.begin()->second->NewComputation();
    std::vector<std::shared_ptr<NodeTree>> trees;
    for (int j = i; j < openings_.size() && j < i + 320; j++) {
      auto tree = std::make_shared<NodeTree>();
      trees.push_back(tree);
      tree->ResetToPosition(openings_[j].start_fen, {});

      for (Move m : openings_[j].moves) {
        tree->MakeMove(m);
      }
      const auto& history = tree->GetPositionHistory();
      if (history.ComputeGameResult() == GameResult::UNDECIDED) {
        int transform;
        auto planes = EncodePositionForNN(
            networks_.begin()->second->GetCapabilities().input_format, history,
            8,
            FillEmptyHistory::FEN_ONLY, &transform);
        comp->AddInput(std::move(planes));
      } 
    }
    comp->ComputeBlocking();
    int compIndex = 0;
    for (int j = i; j < openings_.size() && j < i + 320; j++) {
      auto tree = trees[j - i];
      const auto& history = tree->GetPositionHistory();
      if (history.ComputeGameResult() == GameResult::UNDECIDED) {
        std::cout << "Opening: " << j << " Q: " << comp->GetQVal(compIndex)
                  << " D: " << comp->GetDVal(compIndex) << std::endl;
        compIndex++;
      } else {
        std::cout << "Opening: " << j << " is already decided!!" << std::endl;
      }
    }
  }

}

void SelfPlayTournament::PlayOneGame(int game_number) { return; }

void SelfPlayTournament::Worker() {
  return;
}

void SelfPlayTournament::StartAsync() {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < kParallelism) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void SelfPlayTournament::RunBlocking() {
  if (kParallelism == 1) {
    // No need for multiple threads if there is one worker.
    Worker();
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  } else {
    StartAsync();
    Wait();
  }
}

void SelfPlayTournament::Wait() {
  {
    Mutex::Lock lock(threads_mutex_);
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }
  {
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  }
}

void SelfPlayTournament::Abort() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
  for (auto& game : games_)
    if (game) game->Abort();
}

void SelfPlayTournament::Stop() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
}

SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}

}  // namespace lczero
