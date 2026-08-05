// test_probe_signal_probe.cpp: tests for SignalProbe, NoOpProbe, ProbedStage.
//
// Per CLAUDE.md, tests use `if (!cond) throw std::runtime_error(...)`.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sw/dsp/probe/signal_probe.hpp>

// Portable scratch path. These tests used to hardcode "/tmp/...", which does
// not exist on Windows, so every CSV/JSON round-trip here failed under MSVC
// while passing everywhere else.
static std::string temp_test_path(const char* filename) {
	return (std::filesystem::temp_directory_path() / filename).string();
}

using sw::dsp::probe::SignalProbe;
using sw::dsp::probe::NoOpProbe;
using sw::dsp::probe::ProbedStage;
using sw::dsp::probe::make_probe;

// ---------------------------------------------------------------------------
// Round-trip: push N samples, samples() returns them in chronological order.
// ---------------------------------------------------------------------------
static void test_basic_capture() {
	SignalProbe<double> p("basic", /*capacity=*/8, /*fs=*/1000.0);
	if (p.size() != 0) throw std::runtime_error("initial size must be 0");
	if (p.is_full())   throw std::runtime_error("initial must not be full");

	for (int i = 1; i <= 5; ++i) p.push(static_cast<double>(i));
	if (p.size() != 5) throw std::runtime_error("expected 5 samples");

	auto s = p.samples();
	if (s.size() != 5) throw std::runtime_error("expected span size 5");
	for (std::size_t i = 0; i < s.size(); ++i) {
		if (s[i] != static_cast<double>(i + 1))
			throw std::runtime_error("under-fill order broken");
	}
	if (p.label() != std::string("basic"))
		throw std::runtime_error("label mismatch");
	if (p.sample_rate() != 1000.0)
		throw std::runtime_error("sample_rate mismatch");
}

// ---------------------------------------------------------------------------
// Ring wraparound: push more than capacity; samples() gives the last
// `capacity` values in chronological order.
// ---------------------------------------------------------------------------
static void test_wraparound() {
	SignalProbe<int> p("wrap", /*capacity=*/4, /*fs=*/1.0);
	for (int i = 0; i < 10; ++i) p.push(i);   // 0..9, capacity 4
	if (!p.is_full()) throw std::runtime_error("must be full after 10 pushes");
	if (p.size() != 4) throw std::runtime_error("size must equal capacity");

	auto s = p.samples();
	if (s.size() != 4) throw std::runtime_error("span size must equal capacity");
	// Newest 4 samples in push order: 6, 7, 8, 9.
	for (std::size_t i = 0; i < s.size(); ++i) {
		if (s[i] != static_cast<int>(6 + i))
			throw std::runtime_error("wraparound order broken");
	}
}

// ---------------------------------------------------------------------------
// Reset via clear(): size back to 0, capacity preserved.
// ---------------------------------------------------------------------------
static void test_clear() {
	SignalProbe<double> p("c", 4, 1.0);
	p.push(1.0); p.push(2.0); p.push(3.0);
	p.clear();
	if (p.size() != 0) throw std::runtime_error("clear must zero size");
	if (p.samples().size() != 0) throw std::runtime_error("clear leaves empty span");
}

// ---------------------------------------------------------------------------
// CSV dump + sidecar JSON: file exists, header present, correct number of rows.
// ---------------------------------------------------------------------------
static void test_dump_csv() {
	SignalProbe<double> p("dump_test", 4, 48000.0);
	for (int i = 0; i < 3; ++i) p.push(0.1 * i);
	const std::string csv_path = temp_test_path("_test_probe_signal_probe.csv");
	p.dump_csv(csv_path);

	// Verify CSV contents.
	std::ifstream in(csv_path);
	if (!in) throw std::runtime_error("dump_csv: CSV not created");
	std::string line;
	std::getline(in, line);
	if (line != "sample_index,sample_value")
		throw std::runtime_error("CSV header wrong: " + line);
	int row_count = 0;
	while (std::getline(in, line)) ++row_count;
	if (row_count != 3)
		throw std::runtime_error("CSV row count wrong");

	// Verify sidecar JSON exists + contains label.
	std::ifstream jin(csv_path + ".json");
	if (!jin) throw std::runtime_error("dump_csv: JSON sidecar not created");
	std::stringstream buf;
	buf << jin.rdbuf();
	const std::string json = buf.str();
	if (json.find("dump_test") == std::string::npos)
		throw std::runtime_error("JSON sidecar missing label");
	if (json.find("48000") == std::string::npos)
		throw std::runtime_error("JSON sidecar missing sample_rate");

	// Cleanup - test is idempotent across reruns.
	std::remove(csv_path.c_str());
	std::remove((csv_path + ".json").c_str());
}

// ---------------------------------------------------------------------------
// NoOpProbe: same API, everything is a no-op.
// ---------------------------------------------------------------------------
static void test_noop_probe() {
	NoOpProbe<double> p("np", 8, 1000.0);
	p.push(1.0);
	p.push(2.0);
	p.push(3.0);
	if (p.size() != 0) throw std::runtime_error("NoOpProbe.size() must stay 0");
	if (p.samples().size() != 0)
		throw std::runtime_error("NoOpProbe.samples() must be empty");
	if (p.capacity() != 0) throw std::runtime_error("NoOpProbe.capacity() must be 0");
	if (p.is_full())       throw std::runtime_error("NoOpProbe.is_full() must be false");
	p.clear();
	p.dump_csv(temp_test_path("_should_not_exist.csv"));  // must not throw
}

// ---------------------------------------------------------------------------
// ProbedStage: wrap a trivial stage, verify process() outputs are captured.
// ---------------------------------------------------------------------------
namespace {
struct DoublingStage {
	using sample_scalar = double;
	double process(double x) { return 2.0 * x; }
};

struct DecimatingStage {
	using sample_scalar = double;
	int cnt = 0;
	std::pair<bool, double> process(double x) {
		++cnt;
		if (cnt == 2) { cnt = 0; return {true, x}; }
		return {false, 0.0};
	}
};
} // namespace

static void test_probed_stage_simple() {
	auto p = make_probe(DoublingStage{}, "doubled", 8, 1000.0);
	if (p.process(1.5) != 3.0) throw std::runtime_error("process delegation broken");
	if (p.process(2.5) != 5.0) throw std::runtime_error("process delegation broken");
	auto s = p.probe().samples();
	if (s.size() != 2)   throw std::runtime_error("probe should have 2 samples");
	if (s[0] != 3.0)     throw std::runtime_error("probe sample[0] wrong");
	if (s[1] != 5.0)     throw std::runtime_error("probe sample[1] wrong");
}

// A stage that returns std::pair<bool, T> should only push when bool=true.
static void test_probed_stage_decimating() {
	auto p = make_probe(DecimatingStage{}, "dec", 8, 1000.0);
	p.process(1.0);   // returns {false, 0}; no push
	p.process(2.0);   // returns {true, 2}; push 2.0
	p.process(3.0);   // {false, 0}
	p.process(4.0);   // {true, 4}
	auto s = p.probe().samples();
	if (s.size() != 2) throw std::runtime_error("decimating: expected 2 pushed samples");
	if (s[0] != 2.0)   throw std::runtime_error("decimating sample[0] wrong");
	if (s[1] != 4.0)   throw std::runtime_error("decimating sample[1] wrong");
}

// ---------------------------------------------------------------------------
// Capacity=1 sanity: newest sample only.
// ---------------------------------------------------------------------------
static void test_capacity_one() {
	SignalProbe<double> p("c1", 1, 1.0);
	for (int i = 1; i <= 5; ++i) p.push(static_cast<double>(i));
	auto s = p.samples();
	if (s.size() != 1) throw std::runtime_error("capacity=1 span size wrong");
	if (s[0] != 5.0)   throw std::runtime_error("capacity=1 keeps newest");
}

int main() {
	try {
		std::cout << "test_probe_signal_probe\n";
		test_basic_capture();          std::cout << "  basic_capture       PASS\n";
		test_wraparound();             std::cout << "  wraparound          PASS\n";
		test_clear();                  std::cout << "  clear               PASS\n";
		test_dump_csv();               std::cout << "  dump_csv            PASS\n";
		test_noop_probe();             std::cout << "  noop_probe          PASS\n";
		test_probed_stage_simple();    std::cout << "  probed_stage_simple PASS\n";
		test_probed_stage_decimating();std::cout << "  probed_stage_dec    PASS\n";
		test_capacity_one();           std::cout << "  capacity_one        PASS\n";
		std::cout << "OK\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "FAIL: " << ex.what() << "\n";
		return 1;
	}
}
