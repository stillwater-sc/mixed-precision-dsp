#pragma once
// wav.hpp: WAV file reader/writer
//
// Header-only implementation of the RIFF WAVE format.
// Supports 8-bit, 16-bit, 24-bit, 32-bit integer PCM and 32-bit float PCM.
// Little-endian per the WAV specification.
//
// Copyright (C) 2024-2026 Stillwater Supercomputing, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace sw::dsp::io {

struct WavData {
	std::vector<std::vector<double>> channels;  // channels[ch][sample], normalized to [-1, 1]
	int sample_rate{0};
	int bits_per_sample{0};
	int num_channels{0};

	std::size_t num_samples() const {
		return channels.empty() ? 0 : channels[0].size();
	}
};

namespace detail {

inline void write_le16(std::ostream& os, uint16_t v) {
	char b[2] = { static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF) };
	os.write(b, 2);
}

inline void write_le32(std::ostream& os, uint32_t v) {
	char b[4] = { static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
	              static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF) };
	os.write(b, 4);
}

inline uint16_t read_le16(std::istream& is) {
	uint8_t b[2];
	is.read(reinterpret_cast<char*>(b), 2);
	return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

inline uint32_t read_le32(std::istream& is) {
	uint8_t b[4];
	is.read(reinterpret_cast<char*>(b), 4);
	return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
	       (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

inline double clamp_sample(double v) {
	return std::max(-1.0, std::min(1.0, v));
}

} // namespace detail

// Write a multi-channel WAV file.
template <typename T>
void write_wav_channels(const std::string& path,
                         const std::vector<std::span<const T>>& channels,
                         int sample_rate, int bits_per_sample = 16) {
	if (channels.empty()) throw std::invalid_argument("wav: no channels");

	std::ofstream ofs(path, std::ios::binary);
	if (!ofs) throw std::runtime_error("wav: cannot open " + path + " for writing");

	const int num_channels = static_cast<int>(channels.size());
	const auto num_samples = channels[0].size();
	const int bytes_per_sample = bits_per_sample / 8;
	const uint16_t audio_format = (bits_per_sample == 32 && bytes_per_sample == 4) ? 3 : 1;
	// format 1 = PCM integer, format 3 = IEEE float
	const uint32_t data_size = static_cast<uint32_t>(num_samples * num_channels * bytes_per_sample);
	const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * num_channels * bytes_per_sample);
	const uint16_t block_align = static_cast<uint16_t>(num_channels * bytes_per_sample);

	// RIFF header
	ofs.write("RIFF", 4);
	detail::write_le32(ofs, 36 + data_size);
	ofs.write("WAVE", 4);

	// fmt chunk
	ofs.write("fmt ", 4);
	detail::write_le32(ofs, 16);                          // chunk size
	detail::write_le16(ofs, audio_format);                // PCM=1, float=3
	detail::write_le16(ofs, static_cast<uint16_t>(num_channels));
	detail::write_le32(ofs, static_cast<uint32_t>(sample_rate));
	detail::write_le32(ofs, byte_rate);
	detail::write_le16(ofs, block_align);
	detail::write_le16(ofs, static_cast<uint16_t>(bits_per_sample));

	// data chunk
	ofs.write("data", 4);
	detail::write_le32(ofs, data_size);

	// Interleave and write samples
	for (std::size_t n = 0; n < num_samples; ++n) {
		for (int ch = 0; ch < num_channels; ++ch) {
			double v = (n < channels[ch].size()) ? detail::clamp_sample(static_cast<double>(channels[ch][n])) : 0.0;

			if (bits_per_sample == 8) {
				// 8-bit WAV is unsigned, 0-255, midpoint 128
				uint8_t s = static_cast<uint8_t>((v + 1.0) * 0.5 * 255.0);
				ofs.write(reinterpret_cast<char*>(&s), 1);
			} else if (bits_per_sample == 16) {
				int16_t s = static_cast<int16_t>(v * 32767.0);
				char b[2] = { static_cast<char>(s & 0xFF), static_cast<char>((s >> 8) & 0xFF) };
				ofs.write(b, 2);
			} else if (bits_per_sample == 24) {
				int32_t s = static_cast<int32_t>(v * 8388607.0);
				char b[3] = { static_cast<char>(s & 0xFF), static_cast<char>((s >> 8) & 0xFF),
				              static_cast<char>((s >> 16) & 0xFF) };
				ofs.write(b, 3);
			} else if (bits_per_sample == 32 && audio_format == 1) {
				int32_t s = static_cast<int32_t>(v * 2147483647.0);
				detail::write_le32(ofs, static_cast<uint32_t>(s));
			} else if (bits_per_sample == 32 && audio_format == 3) {
				float f = static_cast<float>(v);
				char b[4];
				std::memcpy(b, &f, 4);
				ofs.write(b, 4);
			}
		}
	}
}

// Write a mono WAV file from any scalar type.
template <typename T>
void write_wav(const std::string& path, std::span<const T> samples,
               int sample_rate, int bits_per_sample = 16) {
	std::vector<std::span<const T>> channels = { samples };
	write_wav_channels(path, channels, sample_rate, bits_per_sample);
}

// Write a stereo WAV file.
template <typename T>
void write_wav(const std::string& path,
               std::span<const T> left, std::span<const T> right,
               int sample_rate, int bits_per_sample = 16) {
	std::vector<std::span<const T>> channels = { left, right };
	write_wav_channels(path, channels, sample_rate, bits_per_sample);
}

// Read a WAV file, returning normalized samples in [-1, 1].
inline WavData read_wav(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) throw std::runtime_error("wav: cannot open " + path + " for reading");

	// RIFF header
	char riff[4];
	ifs.read(riff, 4);
	if (std::strncmp(riff, "RIFF", 4) != 0)
		throw std::runtime_error("wav: not a RIFF file");

	detail::read_le32(ifs);  // file size - 8
	char wave[4];
	ifs.read(wave, 4);
	if (std::strncmp(wave, "WAVE", 4) != 0)
		throw std::runtime_error("wav: not a WAVE file");

	uint16_t audio_format = 0;
	uint16_t num_channels = 0;
	uint32_t sample_rate = 0;
	uint16_t bits_per_sample = 0;
	uint32_t data_size = 0;
	bool found_fmt = false, found_data = false;

	// Read chunks
	while (ifs && !(found_fmt && found_data)) {
		char chunk_id[4];
		ifs.read(chunk_id, 4);
		if (!ifs) break;
		uint32_t chunk_size = detail::read_le32(ifs);

		if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
			audio_format = detail::read_le16(ifs);
			num_channels = detail::read_le16(ifs);
			sample_rate = detail::read_le32(ifs);
			detail::read_le32(ifs);  // byte rate
			detail::read_le16(ifs);  // block align
			bits_per_sample = detail::read_le16(ifs);
			// Skip any extra format bytes
			if (chunk_size > 16) {
				ifs.seekg(chunk_size - 16, std::ios::cur);
			}
			found_fmt = true;
		} else if (std::strncmp(chunk_id, "data", 4) == 0) {
			data_size = chunk_size;
			found_data = true;
			// Don't skip — we'll read the data below
		} else {
			// Skip unknown chunks
			ifs.seekg(chunk_size, std::ios::cur);
		}
	}

	if (!found_fmt || !found_data)
		throw std::runtime_error("wav: missing fmt or data chunk");

	int bytes_per_sample = bits_per_sample / 8;
	std::size_t num_samples = data_size / (num_channels * bytes_per_sample);

	WavData result;
	result.sample_rate = static_cast<int>(sample_rate);
	result.bits_per_sample = static_cast<int>(bits_per_sample);
	result.num_channels = static_cast<int>(num_channels);
	result.channels.resize(num_channels);
	for (auto& ch : result.channels) ch.resize(num_samples);

	for (std::size_t n = 0; n < num_samples; ++n) {
		for (int ch = 0; ch < num_channels; ++ch) {
			double v = 0.0;
			if (bits_per_sample == 8) {
				uint8_t s;
				ifs.read(reinterpret_cast<char*>(&s), 1);
				v = (static_cast<double>(s) / 255.0) * 2.0 - 1.0;
			} else if (bits_per_sample == 16) {
				int16_t s = static_cast<int16_t>(detail::read_le16(ifs));
				v = static_cast<double>(s) / 32767.0;
			} else if (bits_per_sample == 24) {
				uint8_t b[3];
				ifs.read(reinterpret_cast<char*>(b), 3);
				int32_t s = static_cast<int32_t>(b[0]) |
				           (static_cast<int32_t>(b[1]) << 8) |
				           (static_cast<int32_t>(b[2]) << 16);
				if (s & 0x800000) s |= static_cast<int32_t>(0xFF000000u);  // sign extend
				v = static_cast<double>(s) / 8388607.0;
			} else if (bits_per_sample == 32 && audio_format == 1) {
				int32_t s = static_cast<int32_t>(detail::read_le32(ifs));
				v = static_cast<double>(s) / 2147483647.0;
			} else if (bits_per_sample == 32 && audio_format == 3) {
				float f;
				char b[4];
				ifs.read(b, 4);
				std::memcpy(&f, b, 4);
				v = static_cast<double>(f);
			}
			result.channels[ch][n] = v;
		}
	}

	return result;
}

} // namespace sw::dsp::io
