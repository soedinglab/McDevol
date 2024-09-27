#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

bool condition_header(std::ifstream &fasta, std::string &line) {
    if (line[0] == '>') {
        return true;
    }
    else {
        return false;
    }
}

bool line_check(std::ifstream &fasta, std::string &line) {
    if(line.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos) {
        fasta >> line;
        return false;
    }
    else {
        return true;
    }
}

void get_complete_sequence(std::ifstream &fasta, std::string &line, std::string &sequence) {
    fasta >> line;
    while (!condition_header(fasta, line) && !fasta.eof()) {
        if (line_check(fasta, line)) {
            sequence.append(line);
            fasta >> line;
        }
    }
}

void handle_outdir(std::string outdir) {
    try {
        if (!fs::exists(outdir)) {
            if (fs::create_directory(outdir)) {
                std::cout << "Output directory created: " << outdir << "\n";
            } else {
                std::cerr << "Failed to create output directory: " << outdir << "\n";
            }
        } else {
            std::cout << "Output directory already exists! clearing it: " << outdir << "\n";
            for (const auto& entry : fs::directory_iterator(outdir)) {
                if (fs::is_regular_file(entry.path())) {
                    fs::remove(entry.path());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
    
}

int main(int argc, char *argv[]) {

    if(argc != 6 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        std::cout << "input missing !\n";
        std::cout << "usage: ./get_sequence_bybins tmp_dir bin_ids contigs output outdir\n";
        return EXIT_SUCCESS;
    }

    else {
        auto start = std::chrono::high_resolution_clock::now();

        // folder to find bin id file
        std::string tmp_dir = argv[1];
        
        // file with contig id and bin id
        std::string bin_ids = argv[2];
        
        // contig fasta file
        std::string contigs = argv[3];

        std::string output = argv[4];
        std::string outdir = argv[5];

        std::fstream binassignment;
        binassignment.open(tmp_dir + bin_ids);

        if (!binassignment.is_open()) {
            std::cerr << "Error: Unable to open bin assignment file!" << "\n";
            return 1;
        }

        handle_outdir(outdir);

        std::unordered_multimap <std::string, int> bins_ids;
        std::string line, nr, contig_name, bin_id, sequence; 

        while (getline(binassignment, line)) {
            std::istringstream ss(line);
            getline(ss,contig_name,',');
            getline(ss,bin_id,',');
            bins_ids.emplace(contig_name,std::stoi(bin_id));
        }

        std::ifstream fastaFile(contigs);

        // Check if the file was successfully opened
        if (!fastaFile.is_open()) {
            std::cerr << "Error: Unable to open Fasta file!" << "\n";
            return 1;
        }

        fastaFile >> line;
        while (condition_header(fastaFile, line)) {
            sequence = "";
            auto range = bins_ids.equal_range(line.substr(1,-1));
            if (range.first != range.second) {
                get_complete_sequence(fastaFile, line, sequence);
                
                for (auto it = range.first; it != range.second; ++it) {

                    std::string out = std::to_string(it->second);
                    std::string filename = outdir + '/' + output + '_' + out + ".fasta";
                    std::fstream writetofile;
                    writetofile.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);
                    if (!writetofile) {
                        writetofile << ">" << it->first <<"\n" << sequence <<  "\n";
                    }
                    else {
                        writetofile << ">" << it->first <<"\n" << sequence <<  "\n";
                    }
                    writetofile.close();
                }
            }
            else {
                fastaFile >> line;
                while (!condition_header(fastaFile, line) && !fastaFile.eof()) {
                    fastaFile >> line;
                }
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << "completed segregating sequences into corresponding bins in " << duration.count() << " seconds\n";

        return EXIT_SUCCESS;
    }
}
