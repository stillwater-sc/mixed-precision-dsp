// test_io.cpp: test signal file I/O (WAV, CSV, raw binary)
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <sw/dsp/io/io.hpp>
#include <sw/dsp/signals/generators.hpp>

#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace sw::dsp;
using namespace sw::dsp::io;

bool near(double a, double b, double eps) {
	return std::abs(a - b) < eps;
}

// RAII temp file helper
struct TempFile {
	std::string path;
	TempFile(const std::string& name) : path(std::filesystem::temp_directory_path().string() + "/" + name) {}
	~TempFile() noexcept { std::error_code ec; std::filesystem::remove(path, ec); }
};

void test_wav_16bit_roundtrip() {
	TempFile tmp("test_dsp_16bit.wav");
	constexpr int sr = 44100;

	// Generate a 440 Hz sine
	auto sig = sine<double>(1000, 440.0, static_cast<double>(sr));
	std::vector<double> samples(sig.begin(), sig.end());

	// Write
	write_wav<double>(tmp.path, std::span<const double>(samples), sr, 16);

	// Read back
	auto wav = read_wav(tmp.path);
	if (!(wav.sample_rate == sr)) throw std::runtime_error("test failed: wav.sample_rate == sr");
	if (!(wav.bits_per_sample == 16)) throw std::runtime_error("test failed: wav.bits_per_sample == 16");
	if (!(wav.num_channels == 1)) throw std::runtime_error("test failed: wav.num_channels == 1");
	if (!(wav.num_samples() == samples.size())) throw std::runtime_error("test failed: wav.num_samples() == samples.size()");

	// Compare within 16-bit quantization error (~1/32768 = 3e-5)
	double max_err = 0;
	for (std::size_t i = 0; i < samples.size(); ++i) {
		double err = std::abs(wav.channels[0][i] - samples[i]);
		max_err = std::max(max_err, err);
	}
	if (!(max_err < 1e-3)) throw std::runtime_error("test failed: max_err < 1e-3");  // 16-bit: ~3e-5 per sample, some accumulation

	std::cout << "  wav_16bit_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_wav_8bit_roundtrip() {
	TempFile tmp("test_dsp_8bit.wav");
	constexpr int sr = 22050;

	auto sig = sine<double>(500, 440.0, static_cast<double>(sr));
	std::vector<double> samples(sig.begin(), sig.end());

	write_wav<double>(tmp.path, std::span<const double>(samples), sr, 8);

	auto wav = read_wav(tmp.path);
	if (!(wav.sample_rate == sr)) throw std::runtime_error("test failed: wav.sample_rate == sr");
	if (!(wav.bits_per_sample == 8)) throw std::runtime_error("test failed: wav.bits_per_sample == 8");
	if (!(wav.num_samples() == samples.size())) throw std::runtime_error("test failed: wav.num_samples() == samples.size()");

	// 8-bit: ~1/128 = 7.8e-3 quantization
	double max_err = 0;
	for (std::size_t i = 0; i < samples.size(); ++i) {
		max_err = std::max(max_err, std::abs(wav.channels[0][i] - samples[i]));
	}
	if (!(max_err < 0.02)) throw std::runtime_error("test failed: max_err < 0.02");

	std::cout << "  wav_8bit_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_wav_24bit_roundtrip() {
	TempFile tmp("test_dsp_24bit.wav");
	constexpr int sr = 48000;

	auto sig = sine<double>(500, 1000.0, static_cast<double>(sr));
	std::vector<double> samples(sig.begin(), sig.end());

	write_wav<double>(tmp.path, std::span<const double>(samples), sr, 24);

	auto wav = read_wav(tmp.path);
	if (!(wav.sample_rate == sr)) throw std::runtime_error("test failed: wav.sample_rate == sr");
	if (!(wav.bits_per_sample == 24)) throw std::runtime_error("test failed: wav.bits_per_sample == 24");

	double max_err = 0;
	for (std::size_t i = 0; i < samples.size(); ++i) {
		max_err = std::max(max_err, std::abs(wav.channels[0][i] - samples[i]));
	}
	if (!(max_err < 1e-5)) throw std::runtime_error("test failed: max_err < 1e-5");  // 24-bit: ~1.2e-7

	std::cout << "  wav_24bit_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_wav_stereo() {
	TempFile tmp("test_dsp_stereo.wav");
	constexpr int sr = 44100;

	auto left  = sine<double>(500, 440.0, static_cast<double>(sr));
	auto right = sine<double>(500, 880.0, static_cast<double>(sr));
	std::vector<double> l(left.begin(), left.end());
	std::vector<double> r(right.begin(), right.end());

	write_wav<double>(tmp.path, std::span<const double>(l), std::span<const double>(r), sr, 16);

	auto wav = read_wav(tmp.path);
	if (!(wav.num_channels == 2)) throw std::runtime_error("test failed: wav.num_channels == 2");
	if (!(wav.num_samples() == 500)) throw std::runtime_error("test failed: wav.num_samples() == 500");

	// Verify left and right are different
	double diff = 0;
	for (std::size_t i = 0; i < wav.num_samples(); ++i) {
		diff += std::abs(wav.channels[0][i] - wav.channels[1][i]);
	}
	if (!(diff > 1.0)) throw std::runtime_error("test failed: diff > 1.0");  // they should differ significantly

	std::cout << "  wav_stereo: passed\n";
}

void test_wav_32bit_roundtrip() {
	TempFile tmp("test_dsp_32bit.wav");
	constexpr int sr = 48000;

	auto sig = sine<double>(500, 1000.0, static_cast<double>(sr));
	std::vector<double> samples(sig.begin(), sig.end());

	write_wav<double>(tmp.path, std::span<const double>(samples), sr, 32);

	auto wav = read_wav(tmp.path);
	if (!(wav.sample_rate == sr)) throw std::runtime_error("test failed: wav.sample_rate == sr");
	if (!(wav.bits_per_sample == 32)) throw std::runtime_error("test failed: wav.bits_per_sample == 32");

	double max_err = 0;
	for (std::size_t i = 0; i < samples.size(); ++i) {
		max_err = std::max(max_err, std::abs(wav.channels[0][i] - samples[i]));
	}
	if (!(max_err < 1e-8)) throw std::runtime_error("test failed: max_err < 1e-8");  // 32-bit int: ~4.7e-10

	std::cout << "  wav_32bit_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_csv_roundtrip() {
	TempFile tmp("test_dsp.csv");

	auto sig = sine<double>(100, 10.0, 1000.0);
	std::vector<double> samples(sig.begin(), sig.end());

	write_csv<double>(tmp.path, std::span<const double>(samples), "signal");

	auto read_back = read_csv<double>(tmp.path, 0);
	if (!(read_back.size() == samples.size())) throw std::runtime_error("test failed: read_back.size() == samples.size()");

	double max_err = 0;
	for (std::size_t i = 0; i < samples.size(); ++i) {
		max_err = std::max(max_err, std::abs(read_back[i] - samples[i]));
	}
	if (!(max_err < 1e-10)) throw std::runtime_error("test failed: max_err < 1e-10");  // text format preserves 15 digits

	std::cout << "  csv_roundtrip: passed (max_err=" << max_err << ")\n";
}

void test_csv_two_column() {
	TempFile tmp("test_dsp_2col.csv");

	std::vector<double> time = {0.0, 0.1, 0.2, 0.3, 0.4};
	std::vector<double> signal = {0.0, 0.5, 1.0, 0.5, 0.0};

	write_csv<double, double>(tmp.path,
		std::span<const double>(time),
		std::span<const double>(signal),
		"time", "amplitude");

	auto col0 = read_csv<double>(tmp.path, 0);
	auto col1 = read_csv<double>(tmp.path, 1);

	if (!(col0.size() == 5)) throw std::runtime_error("test failed: col0.size() == 5");
	if (!(col1.size() == 5)) throw std::runtime_error("test failed: col1.size() == 5");
	if (!(near(col0[2], 0.2, 1e-10))) throw std::runtime_error("test failed: near(col0[2], 0.2, 1e-10)");
	if (!(near(col1[2], 1.0, 1e-10))) throw std::runtime_error("test failed: near(col1[2], 1.0, 1e-10)");

	std::cout << "  csv_two_column: passed\n";
}

void test_raw_roundtrip() {
	TempFile tmp("test_dsp.raw");

	std::vector<double> samples = {0.1, 0.2, 0.3, -0.5, 0.99};
	write_raw<double>(tmp.path, std::span<const double>(samples));

	auto read_back = read_raw<double>(tmp.path, 5);
	if (!(read_back.size() == 5)) throw std::runtime_error("test failed: read_back.size() == 5");

	for (std::size_t i = 0; i < samples.size(); ++i) {
		if (!(read_back[i] == samples[i])) throw std::runtime_error("test failed: read_back[i] == samples[i]");  // exact binary match
	}

	std::cout << "  raw_roundtrip: passed\n";
}

void test_raw_read_all() {
	TempFile tmp("test_dsp_all.raw");

	std::vector<float> samples = {1.0f, 2.0f, 3.0f};
	write_raw<float>(tmp.path, std::span<const float>(samples));

	auto read_back = read_raw<float>(tmp.path);
	if (!(read_back.size() == 3)) throw std::runtime_error("test failed: read_back.size() == 3");
	if (!(read_back[0] == 1.0f)) throw std::runtime_error("test failed: read_back[0] == 1.0f");
	if (!(read_back[2] == 3.0f)) throw std::runtime_error("test failed: read_back[2] == 3.0f");

	std::cout << "  raw_read_all: passed\n";
}

int main() {
	try {
		std::cout << "Signal File I/O Tests\n";

		test_wav_16bit_roundtrip();
		test_wav_8bit_roundtrip();
		test_wav_24bit_roundtrip();
		test_wav_32bit_roundtrip();
		test_wav_stereo();
		test_csv_roundtrip();
		test_csv_two_column();
		test_raw_roundtrip();
		test_raw_read_all();

		std::cout << "All signal I/O tests passed.\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "FAILED: " << e.what() << '\n';
		return 1;
	} catch (...) {
		std::cerr << "FAILED: unknown exception\n";
		return 1;
	}
}
