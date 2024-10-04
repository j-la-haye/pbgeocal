#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <cstring>

using data_t = uint16_t;

constexpr int64_t aviris4img_channels = 327; //this does not include the band with time tags
constexpr int64_t aviris4img_resolution = 1280;
constexpr int64_t aviris4img_headerlinelen = aviris4img_resolution*sizeof (data_t);
constexpr int64_t aviris4img_linelen = aviris4img_resolution*(aviris4img_channels+1)*sizeof (data_t);
constexpr int64_t aviris4img_linedatalen = aviris4img_resolution*aviris4img_channels*sizeof (data_t);

constexpr int64_t sysTimeOffset = 0;
constexpr int64_t statusFlagOffset = 80;
constexpr int64_t statusFlagExpected = 0xBABE;
constexpr int64_t utcTowOffset = 116;
constexpr int64_t sysTimePPSOffset = 164;

std::vector<int64_t> loadFrameTimes(std::string const& frameFilePath) {

    std::filesystem::path p(frameFilePath);
    int64_t fileSize = std::filesystem::file_size(p);

    std::ifstream file(frameFilePath);

    if (!file.is_open()) {
        return std::vector<int64_t>();
    }

    if (fileSize % aviris4img_linelen != 0) { //unexpected size
        return std::vector<int64_t>();
    }

    int nLines = fileSize/aviris4img_linelen;

    struct lineTimingInfos {
        int64_t internalTime;
        int64_t gpsTimeLastPPS;
        int64_t internalTimeLastPPS;
        bool isBabe;
    };

    std::vector<lineTimingInfos> infos(nLines);
    std::vector<int64_t> ret(nLines);

    char* headerData = new char[aviris4img_headerlinelen];

    for (int i = 0; i < nLines; i++) {

        file.seekg(i*aviris4img_linelen);
        file.read(headerData, aviris4img_headerlinelen); //first 4 bytes are the time

        uint8_t bytesBuffer[4];

        std::memcpy(bytesBuffer, &(headerData[sysTimeOffset]), 4);
        uint32_t lineInternalTime = bytesBuffer[0] | bytesBuffer[1] << 8 | bytesBuffer[2] << 16 | bytesBuffer[3] << 24;

        std::memcpy(bytesBuffer, &(headerData[statusFlagOffset]), 4);

        uint32_t flag = bytesBuffer[0] | bytesBuffer[1] << 8;
        //BABE if PPS changes

        bool isBabe = false;

        if ((flag ^ 0xBABE) == 0) {
            isBabe = true;
        }

        std::memcpy(bytesBuffer, &(headerData[utcTowOffset]), 4);

        //Big endian
        uint32_t gpsValidityTime = bytesBuffer[3] | bytesBuffer[2] << 8 | bytesBuffer[1] << 16 | bytesBuffer[0] << 24;

        std::memcpy(bytesBuffer, &(headerData[sysTimePPSOffset]), 4);
        uint32_t ppsInternalTime = bytesBuffer[2] | bytesBuffer[3] << 8 | bytesBuffer[0] << 16 | bytesBuffer[1] << 24;

        infos[i] = {lineInternalTime, gpsValidityTime, ppsInternalTime, isBabe};

    }

    delete [] headerData;

    //fill in missing values
    std::vector<size_t> babeIdxs;
    for (int i = 0; i < infos.size(); i++) {
        if (infos[i].isBabe) {
            babeIdxs.push_back(i);
        }
    }

    if (babeIdxs.empty()) {
        return std::vector<int64_t>();
    }

    int previousBabeIdx = babeIdxs[0];
    int nextBabeIdx = babeIdxs[0];
    int currentBabeIdxPos = 0;

    for (int i = 0; i < infos.size(); i++) {
        int delta_prev = std::abs(i - previousBabeIdx);
        int delta_next = std::abs(i - nextBabeIdx);

        if (delta_prev < delta_next) {
            infos[i].gpsTimeLastPPS = infos[previousBabeIdx].gpsTimeLastPPS;
            infos[i].internalTimeLastPPS = infos[previousBabeIdx].internalTimeLastPPS;
        } else {
            infos[i].gpsTimeLastPPS = infos[nextBabeIdx].gpsTimeLastPPS;
            infos[i].internalTimeLastPPS = infos[nextBabeIdx].internalTimeLastPPS;
        }

        if (i == nextBabeIdx) {
            previousBabeIdx = nextBabeIdx;
            currentBabeIdxPos++;
            if (currentBabeIdxPos >= babeIdxs.size()) {
                currentBabeIdxPos = babeIdxs.size()-1;
            }
            nextBabeIdx = babeIdxs[currentBabeIdxPos];
        }
    }

    //at that point infos has been filled such that a gps time reference and internal time for the corresponding pps is set
    for (int i = 0; i < infos.size(); i++) {
        int64_t delta_t = infos[i].internalTime - infos[i].internalTimeLastPPS;

        ret[i] = infos[i].gpsTimeLastPPS*10 + delta_t;
    }

    return ret;

}

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Wrong number of arguments provided, only the input file is required!" << std::endl;
        return 1;
    }

    std::string filePath(argv[1]);
    std::filesystem::path p(filePath);

    if (!std::filesystem::exists(p)) {
        std::cerr << "Provided input file: '" << filePath << "' does not exists!" << std::endl;
        return 1;
    }

    std::vector<int64_t> times = loadFrameTimes(filePath);

    if (times.empty()) {
        std::cerr << "Could not load time data from file: '" << filePath << "'!" << std::endl;
        return 1;
    }

    for(int64_t time : times) {
        std::cout << time << "\n";
    }

    std::cout << std::flush;

    return 0;
}
