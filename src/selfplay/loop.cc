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

#include "selfplay/loop.h"

#include <optional>
#include <sstream>

#include "gtb-probe.h"
#include "chess/pgn.h"
#include "neural/decoder.h"
#include "selfplay/tournament.h"
#include "trainingdata/reader.h"
#include "utils/configfile.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {

namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kGaviotaTablebaseId{"gaviotatb-paths", "",
                                   "List of Gaviota tablebase directories"};
const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kPolicySubsDirId{"policy-substitutions", "",
                                "Directory with gzipped files are to use to "
                                "replace policy for some of the data."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with.", 't'};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kMinDTZBoostId{
    "dtz_policy_boost", "",
    "Additional offset to apply to policy target before temperature for moves "
    "that are best dtz option."};
const OptionId kNewInputFormatId{
    "new-input-format", "",
    "Input format to convert training data to during rescoring."};
const OptionId kDeblunder{
    "deblunder", "",
    "If true, whether to use move Q information to infer a different Z value "
    "if the the selected move appears to be a blunder."};
const OptionId kDeblunderQBlunderThreshold{
    "deblunder-q-blunder-threshold", "",
    "The amount Q of played move needs to be worse than best move in order to "
    "assume the played move is a blunder."};
const OptionId kDeblunderQBlunderWidth{
    "deblunder-q-blunder-width", "",
    "Width of the transition between accepted temp moves and blunders."};
const OptionId kNnuePlainFileId{"nnue-plain-file", "",
                                "Append SF plain format training data to this "
                                "file. Will be generated if not there."};
const OptionId kNnueBestScoreId{"nnue-best-score", "",
                                "For the SF training data use the score of the "
                                "best move instead of the played one."};
const OptionId kNnueBestMoveId{
    "nnue-best-move", "",
    "For the SF training data record the best move instead of the played one. "
    "If set to true the generated files do not compress well."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};

const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console."};

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 1858; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[1858];
  PolicySubNode* children[1858];
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];
std::atomic<int> policy_dtm_bump(0);
std::atomic<int> gaviota_dtm_rescores(0);
std::map<uint64_t, PolicySubNode> policy_subs;
bool gaviotaEnabled = false;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}


void gaviota_tb_probe_hard(const Position& pos, unsigned int& info,
                           unsigned int& dtm) {
  unsigned int wsq[17];
  unsigned int bsq[17];
  unsigned char wpc[17];
  unsigned char bpc[17];

  auto stm = pos.IsBlackToMove() ? tb_BLACK_TO_MOVE : tb_WHITE_TO_MOVE;
  auto& board = pos.IsBlackToMove() ? pos.GetThemBoard() : pos.GetBoard();
  auto epsq = tb_NOSQUARE;
  for (auto sq : board.en_passant()) {
    // Our internal representation stores en_passant 2 rows away
    // from the actual sq.
    if (sq.row() == 0) {
      epsq = (TB_squares)(sq.as_int() + 16);
    } else {
      epsq = (TB_squares)(sq.as_int() - 16);
    }
  }
  int idx = 0;
  for (auto sq : (board.ours() & board.kings())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.ours() & board.knights())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.ours() & board.queens())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.ours() & board.rooks())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.ours() & board.bishops())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.ours() & board.pawns())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_PAWN;
    idx++;
  }
  wsq[idx] = tb_NOSQUARE;
  wpc[idx] = tb_NOPIECE;

  idx = 0;
  for (auto sq : (board.theirs() & board.kings())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.theirs() & board.knights())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.theirs() & board.queens())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.theirs() & board.rooks())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.theirs() & board.bishops())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.theirs() & board.pawns())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_PAWN;
    idx++;
  }
  bsq[idx] = tb_NOSQUARE;
  bpc[idx] = tb_NOPIECE;

  tb_probe_hard(stm, epsq, tb_NOCASTLE, wsq, bsq, wpc, bpc, &info, &dtm);
}


struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

void ProcessFile(const std::string& file, 
                 std::string outputDir, ProcessFileFlags flags) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      PgnReader reader;
      reader.AddPgnFile(file);
      auto data = reader.ReleaseGames();
      std::string fileName = file.substr(file.find_last_of("/\\") + 1);
      TrainingDataWriter writer(outputDir + "/" + fileName);
      for (auto game : data) {
        PositionHistory history;
        ChessBoard board;
        int rule50_ply;
        int move_count;
        board.SetFromFen(game.start_fen, &rule50_ply, &move_count);
        history.Reset(board, rule50_ply, move_count);
        for (auto move : game.moves) {
          history.Append(move);
        }
        V6TrainingData data;
        data.version = 1;
        data.input_format = 1;
        int transform;
        InputPlanes planes = EncodePositionForNN(
            (pblczero::NetworkFormat::InputFormat)1, history, 8,
            FillEmptyHistory::NO, &transform);
        int plane_idx = 0;
        for (auto& plane : data.planes) {
          plane = ReverseBitsInBytes(planes[plane_idx++].mask);
        }
        data.result_q = (float)game.outcome;
        data.result_d = game.outcome == 0 ? 1.0f : 0.0f;
        // This assumes a lot...
        data.p1idx = std::stoi(game.player1.substr(11));
        data.p2idx = std::stoi(game.player2.substr(11));
        writer.WriteChunk(data);
      }
    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
    }
  }
}

void ProcessFiles(const std::vector<std::string>& files,std::string outputDir,
                  int offset, int mod, ProcessFileFlags flags) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (int i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".pgn") != files[i].size() - 4) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], outputDir, flags);
  }
}

}  // namespace

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

void RescoreLoop::RunLoop() {
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  for (int i = 0; i < 11; i++) policy_bump_total_hist[i] = 0;
  for (int i = 0; i < 11; i++) policy_nobump_total_hist[i] = 0;
  options_.Add<StringOption>(kSyzygyTablebaseId);
  options_.Add<StringOption>(kGaviotaTablebaseId);
  options_.Add<StringOption>(kInputDirId);
  options_.Add<StringOption>(kOutputDirId);
  options_.Add<StringOption>(kPolicySubsDirId);
  options_.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options_.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options_.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options_.Add<FloatOption>(kMinDTZBoostId, 0, 1) = 0;
  options_.Add<IntOption>(kNewInputFormatId, -1, 256) = -1;
  options_.Add<BoolOption>(kDeblunder) = false;
  options_.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options_.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options_.Add<StringOption>(kNnuePlainFileId);
  options_.Add<BoolOption>(kNnueBestScoreId) = true;
  options_.Add<BoolOption>(kNnueBestMoveId) = false;
  options_.Add<BoolOption>(kDeleteFilesId) = true;

  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;

  if (options_.GetOptionsDict().IsDefault<std::string>(kOutputDirId)) {
    std::cerr << "Must provide an output dir." << std::endl;
    return;
  }

  auto inputDir = options_.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.size() == 0) {
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (int i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  int threads = options_.GetOptionsDict().Get<int>(kThreadsId);
  ProcessFileFlags flags;
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([this, offset_val, files, threads,
                             flags]() {
        ProcessFiles(
            files,
            options_.GetOptionsDict().Get<std::string>(kOutputDirId),
            offset_val, threads,
            flags);
      });
    }
    for (int i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(files,
                 options_.GetOptionsDict().Get<std::string>(kOutputDirId),
        0, 1,
                 flags);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Rescores performed: " << rescored << std::endl;
  std::cout << "Cumulative outcome change: " << delta << std::endl;
  std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  std::cout << "Secondary rescores performed used dtz: " << rescored3
            << std::endl;
  std::cout << "Blunders picked up by deblunder threshold: " << blunders
            << std::endl;
  std::cout << "Number of policy values boosted by dtz or dtm " << policy_bump
            << std::endl;
  std::cout << "Number of policy values boosted by dtm " << policy_dtm_bump
            << std::endl;
  std::cout << "Orig policy_sum dist of boost candidate:";
  std::cout << std::endl;
  int event_sum = 0;
  for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i];
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_nobump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Boosted policy_sum dist of boost candidate:";
  std::cout << std::endl;
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_bump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
            << " W: " << orig_counts[2] << std::endl;
  std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
            << " W: " << fixed_counts[2] << std::endl;
  std::cout << "Gaviota DTM move_count rescores: " << gaviota_dtm_rescores
            << std::endl;
}

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<BoolOption>(kInteractiveId) = false;
  options_.Add<StringOption>(kLogFileId);

  if (!options_.ProcessAllFlags()) return;

  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));

  if (options_.GetOptionsDict().Get<bool>(kInteractiveId)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetUciOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];

  // Initialize variables.
  float percentage = -1;
  std::optional<float> elo;
  std::optional<float> los;

  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}

}  // namespace lczero
