#include "../libstfio/stfio.h"
#include "../libstfio/biosig/biosiglib.h"
#include "../libstfio/cfs/cfslib.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>

namespace {

// Write a small ABF2 file independently of either reader. Each sweep contains
// eight samples per channel, with different gains, offsets, names and units.
std::vector<std::uint8_t> MakeABF2(int channels, bool floatingPoint) {
    std::vector<std::uint8_t> bytes(6 * 512, 0);
    const auto integer = [&](std::size_t offset, std::uint64_t value, int size) {
        for (int n = 0; n < size; ++n)
            bytes.at(offset + n) = static_cast<std::uint8_t>(value >> (8 * n));
    };
    const auto real = [&](std::size_t offset, float value) {
        std::uint32_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        integer(offset, bits, 4);
    };
    const auto section = [&](int index, int block, int size, int entries) {
        const std::size_t offset = 76 + 16 * index;
        integer(offset, block, 4);
        integer(offset + 4, size, 4);
        integer(offset + 8, entries, 8);
    };

    const int sampleSize = floatingPoint ? 4 : 2;
    const int sweepSamples = 8 * channels;
    // File info and section directory.
    integer(0, 0x32464241, 4);  // ABF2
    integer(4, 0x02090000, 4);  // file version 2.9
    integer(8, 512, 4);
    integer(12, 2, 4);         // two sweeps
    integer(16, 20260101, 4);
    integer(28, 1, 2);         // ABF file type
    integer(30, floatingPoint ? 1 : 0, 2);
    integer(60, 0x0a000000, 4);
    integer(64, 1, 4);         // creator name string index
    section(0, 1, 512, 1);     // protocol
    section(1, 2, 128, channels);
    section(10, 4, sampleSize, 2 * sweepSamples);
    section(15, 5, 8, 2);      // sweep synchronization array

    // Protocol: episodic stimulation, 100 us per sample, integer calibration.
    integer(512, 5, 2);
    real(512 + 2, 100.0F);
    integer(512 + 10, 1, 4);   // compression ratio
    integer(512 + 22, sweepSamples, 4);
    integer(512 + 30, 2, 4);   // episodes per run
    integer(512 + 34, 1, 4);
    integer(512 + 38, 1, 4);
    real(512 + 110, 8.0F);     // ADC range
    integer(512 + 118, 32768, 4);

    for (int channel = 0; channel < channels; ++channel) {
        const std::size_t offset = 1024 + 128 * channel;
        integer(offset, channel, 2);
        real(offset + 6, 1.0F);    // telegraph gain
        integer(offset + 24, channel, 2);
        integer(offset + 26, channel, 2);
        real(offset + 28, 2.0F);   // programmable gain
        real(offset + 32, 1.0F);   // display amplification
        real(offset + 40, channel == 0 ? 0.125F : 0.015625F);
        real(offset + 44, channel == 0 ? 1.0F : -2.0F);
        real(offset + 48, 1.0F);   // signal gain
        integer(offset + 74, 2 + 2 * channel, 4);
        integer(offset + 78, 3 + 2 * channel, 4);
    }

    const char strings[] = "Clampex\0Vm_prime\0mV\0Im_sec\0pA\0";
    section(9, 3, 44 + sizeof(strings) - 1, 1);
    integer(1536, 0x48435353, 4);   // SSCH string cache
    integer(1536 + 4, 1, 4);
    integer(1536 + 8, 5, 4);
    integer(1536 + 12, 8, 4);
    integer(1536 + 16, sizeof(strings) - 1, 4);
    std::memcpy(bytes.data() + 1536 + 44, strings, sizeof(strings) - 1);

    std::size_t offset = 2048;
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (int sample = 0; sample < 8; ++sample) {
            for (int channel = 0; channel < channels; ++channel) {
                const int raw = channel == 0 ? -1000 + 100 * sweep + 10 * sample
                                             : 20 + 10 * sweep + sample;
                if (floatingPoint)
                    real(offset, static_cast<float>(raw) / 8.0F);
                else
                    integer(offset, static_cast<std::uint16_t>(raw), 2);
                offset += sampleSize;
            }
        }
        integer(2560 + 8 * sweep, sweep * sweepSamples, 4);
        integer(2560 + 8 * sweep + 4, sweepSamples, 4);
    }
    return bytes;
}

class ABF2ImportTest : public testing::TestWithParam<std::tuple<int, bool, stfio::filetype>> {
protected:
    std::filesystem::path path;

    void SetUp() override {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path()
            / ("stimfit-abf2-" + std::to_string(unique) + ".abf");
        const auto bytes = MakeABF2(std::get<0>(GetParam()), std::get<1>(GetParam()));
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(file.good());
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(path, error);
        EXPECT_FALSE(error) << error.message();
    }
};

TEST_P(ABF2ImportTest, PreservesSamplesCalibrationAndMetadata) {
    const auto [channels, floatingPoint, type] = GetParam();
    stfio::StdoutProgressInfo progress("ABF import", "", 100, false);
    Recording recording;
    ASSERT_TRUE(stfio::importFile(path.string(), type, recording, stfio::txtImportSettings(), progress));
    ASSERT_EQ(recording.size(), static_cast<std::size_t>(channels));
    EXPECT_DOUBLE_EQ(recording.GetXScale(), 0.1);
    for (int channel = 0; channel < channels; ++channel) {
        const auto& data = recording[channel];
        EXPECT_EQ(data.GetChannelName(), channel == 0 ? "Vm_prime" : "Im_sec");
        EXPECT_EQ(data.GetYUnits(), channel == 0 ? "mV" : "pA");
        ASSERT_EQ(data.size(), std::size_t{2});
        for (int sweep = 0; sweep < 2; ++sweep) {
            ASSERT_EQ(data[sweep].size(), std::size_t{8});
            for (int sample = 0; sample < 8; ++sample) {
                const int raw = channel == 0 ? -1000 + 100 * sweep + 10 * sample
                                             : 20 + 10 * sweep + sample;
                const double expected = floatingPoint ? raw / 8.0
                    : channel == 0 ? raw / 1024.0 + 1.0 : raw / 128.0 - 2.0;
                EXPECT_DOUBLE_EQ(data[sweep][sample], expected)
                    << "channel=" << channel << " sweep=" << sweep << " sample=" << sample;
            }
        }
    }
}

TEST_P(ABF2ImportTest, ExplicitNativeChoiceIsHonoured) {
    stfio::StdoutProgressInfo progress("ABF import", "", 100, false);
    Recording recording;
    auto used = stfio::fileReader::automatic;
    ASSERT_TRUE(stfio::importFile(path.string(), std::get<2>(GetParam()), recording,
        stfio::txtImportSettings(), progress, stfio::fileReader::native, &used));
    EXPECT_EQ(used, stfio::fileReader::native);
    EXPECT_EQ(recording[0].GetChannelName(), "Vm_prime");
    EXPECT_EQ(recording[0][0].size(), std::size_t{8});
}

#ifdef WITH_BIOSIG
TEST_P(ABF2ImportTest, ExplicitBioSigChoiceIsHonoured) {
    stfio::StdoutProgressInfo progress("ABF import", "", 100, false);
    Recording expected, recording;
    ASSERT_EQ(stfio::importBiosigFile(path.string(), expected, progress, true), stfio::biosig);
    auto used = stfio::fileReader::automatic;
    ASSERT_TRUE(stfio::importFile(path.string(), std::get<2>(GetParam()), recording,
        stfio::txtImportSettings(), progress, stfio::fileReader::biosig, &used));
    EXPECT_EQ(used, stfio::fileReader::biosig);
    ASSERT_EQ(recording.size(), expected.size());
    for (std::size_t channel = 0; channel < expected.size(); ++channel) {
        EXPECT_EQ(recording[channel].GetChannelName(), expected[channel].GetChannelName());
        EXPECT_EQ(recording[channel].GetYUnits(), expected[channel].GetYUnits());
        ASSERT_EQ(recording[channel].size(), expected[channel].size());
        for (std::size_t sweep = 0; sweep < expected[channel].size(); ++sweep) {
            ASSERT_EQ(recording[channel][sweep].size(), expected[channel][sweep].size());
            for (std::size_t sample = 0; sample < expected[channel][sweep].size(); ++sample) {
                if (std::isnan(expected[channel][sweep][sample]))
                    EXPECT_TRUE(std::isnan(recording[channel][sweep][sample]));
                else
                    EXPECT_DOUBLE_EQ(recording[channel][sweep][sample], expected[channel][sweep][sample]);
            }
        }
    }
}
#endif

TEST(ReaderSelection, RejectsUnknownReaderNames) {
    EXPECT_EQ(stfio::parseReader("auto"), stfio::fileReader::automatic);
    EXPECT_EQ(stfio::parseReader("biosig"), stfio::fileReader::biosig);
    EXPECT_EQ(stfio::parseReader("native"), stfio::fileReader::native);
    EXPECT_THROW(stfio::parseReader("unknown"), std::invalid_argument);
}

#ifdef WITH_BIOSIG
TEST(ReaderSelection, PrefersBioSigWhenNativeIsAlsoAvailable) {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    struct Fixture {
        std::filesystem::path path;
        ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    } fixture{std::filesystem::temp_directory_path() / ("stimfit-reader-" + std::to_string(unique) + ".cfs")};
    stfio::StdoutProgressInfo progress("CFS import", "", 100, false);
    Recording source(1, 1, 16), result;
    source.SetXScale(0.1);
    source.SetXUnits("ms");
    source[0].SetChannelName("Voltage");
    source[0].SetYUnits("mV");
    for (int n = 0; n < 16; ++n) source[0][0].get_w()[n] = n - 8;
    ASSERT_TRUE(stfio::exportCFSFile(fixture.path.string(), source, progress));
    const auto info = stfio::getReaderInfo(fixture.path.string());
    EXPECT_TRUE(info.biosigAvailable);
    EXPECT_TRUE(info.nativeAvailable);
    EXPECT_EQ(info.recommended, stfio::fileReader::biosig);
    auto used = stfio::fileReader::automatic;
    ASSERT_TRUE(stfio::importFile(fixture.path.string(), stfio::cfs, result,
        stfio::txtImportSettings(), progress, stfio::fileReader::automatic, &used));
    EXPECT_EQ(used, stfio::fileReader::biosig);
    ASSERT_EQ(result[0][0].size(), std::size_t{16});
    EXPECT_DOUBLE_EQ(result[0][0][0], -8.0);
}

TEST(ReaderSelection, PrefersBioSigForOrdinaryFiles) {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    struct Fixture {
        std::filesystem::path path;
        ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    } fixture{std::filesystem::temp_directory_path() / ("stimfit-reader-" + std::to_string(unique) + ".edf")};
    std::string header;
    const auto field = [&](const std::string& value, std::size_t width) {
        header += value + std::string(width - value.size(), ' ');
    };
    field("0", 8); field("Synthetic reader test", 80); field("Known signal", 80);
    field("01.01.26", 8); field("00.00.00", 8); field("512", 8); field("", 44);
    field("1", 8); field("1", 8); field("1", 4);
    field("Signal", 16); field("", 80); field("uV", 8);
    field("-32768", 8); field("32767", 8); field("-32768", 8); field("32767", 8);
    field("", 80); field("16", 8); field("", 32);
    ASSERT_EQ(header.size(), std::size_t{512});
    {
        std::ofstream file(fixture.path, std::ios::binary);
        file.write(header.data(), header.size());
        for (int n = 0; n < 16; ++n) {
            const auto sample = static_cast<std::uint16_t>(n - 8);
            file.put(static_cast<char>(sample & 0xff));
            file.put(static_cast<char>(sample >> 8));
        }
        ASSERT_TRUE(file.good());
    }
    const auto info = stfio::getReaderInfo(fixture.path.string());
    EXPECT_TRUE(info.biosigAvailable);
    EXPECT_EQ(info.recommended, stfio::fileReader::biosig);
    stfio::StdoutProgressInfo progress("EDF import", "", 100, false);
    Recording data;
    auto used = stfio::fileReader::automatic;
    ASSERT_TRUE(stfio::importFile(fixture.path.string(), stfio::none, data,
        stfio::txtImportSettings(), progress, stfio::fileReader::automatic, &used));
    EXPECT_EQ(used, stfio::fileReader::biosig);
    ASSERT_EQ(data[0][0].size(), std::size_t{16});
    for (int n = 0; n < 16; ++n)
        EXPECT_NEAR(data[0][0][n], (n - 8) / 1000.0, 1e-12);
    EXPECT_THROW(stfio::importFile(fixture.path.string(), stfio::none, data,
        stfio::txtImportSettings(), progress, stfio::fileReader::native), std::runtime_error);
}
#endif

INSTANTIATE_TEST_SUITE_P(ABF2, ABF2ImportTest, testing::Combine(
    testing::Values(1, 2), testing::Bool(),
#ifdef WITH_BIOSIG
    testing::Values(stfio::abf, stfio::none)
#else
    testing::Values(stfio::abf)
#endif
));

} // namespace
