#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <filesystem>


namespace fs = std::filesystem;
bool condition_header(std::string &line) {
    if (line[0] == '>') {
        return true;
    }
    else {
        return false;
    }
}

void get_complete_sequence(std::ifstream &fasta, std::string &line, std::string &sequence) {
    getline(fasta, line);
    while (!condition_header(line) && !fasta.eof()) {
        sequence.append(line);
        getline(fasta, line);
    }
}

int main(int argc, char *argv[]) {

    if(argc < 3 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)
    {
        std::cout << "Usage: convertfasta_multi2single input.fasta outdir min_length (optional, int) --length (print length, optional) \n";
        return EXIT_SUCCESS;
    }
    
    else {
        auto start = std::chrono::high_resolution_clock::now();
        std::string contigs_sequences = argv[1];
        auto pos = contigs_sequences.find_last_of ('/');
        std::string outname;
        if (pos != std::string::npos) {
            outname = contigs_sequences.substr(pos+1);
        } else {
            outname = contigs_sequences;
        }
    
        std::string dir = argv[2];
        if (!fs::exists(dir)) {
            if (!fs::create_directories(dir)) {
                std::cerr << "Error: Unable to create output directory!" << "\n";
                return 1;
            }
        }
        unsigned int min_length = 0;
        bool return_length = 0;
        std::ofstream length_file;
        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--length") == 0) {
                return_length = true;
                length_file.open(dir +"/contig_length");
            } else {
                min_length = std::stoi(argv[i]);
            }
        }

        std::ifstream fasta(contigs_sequences);
        std::cout<<"Input file: " << contigs_sequences << "\n";
        std::cout<<"Output directory: " << dir << "\n";
        std::ofstream outfile(dir + "/single_" + outname);

        if (!fasta.is_open()) {
            std::cerr << "Error: Unable to open Fasta file!" << "\n";
            return 1;
        }

        std::string line, sequence;

        getline(fasta, line);
        std::string header;
        unsigned int sequence_counter = 0;
        while (condition_header(line)) {
            header = line;
           
            // Find the start position, which is after '>'
            size_t start_pos = line.find('>') + 1;

            // Find the end position, which is the first space after the start position
            size_t end_pos = line.find(' ', start_pos);

            // Extract the substring
            std::string seq_id = line.substr(start_pos, end_pos - start_pos);
            sequence = "";
            get_complete_sequence(fasta, line, sequence);

            if (!sequence.empty()) {
                if (sequence.length() >= min_length) {
                    outfile << header << "\n";
                    outfile << sequence << "\n";
                    if (return_length==1) {
                        length_file << seq_id << "\t" << sequence.length() << "\n";
                    }
                }
            }
            else {
                std::cerr << "empty sequence for a header" << "\n";
                return 1;
            }
            sequence_counter++;
        }
        outfile.close();
        if (return_length && length_file.is_open()) {
            length_file.close();
        }
        std::cout << sequence_counter << " sequences processed \n";
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << duration.count() << " seconds\n";
        return  EXIT_SUCCESS;
    }
}