#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cmath>
#include <fstream>
#include <cstring>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

const unsigned int FLAG_0 = 0;      // 0 primary alignment
const unsigned int FLAG_1 = 1;      // 0x1 template having multiple segments in sequencing
const unsigned int PAIRED_FLAG = 2;      // 0x2 each segment properly aligned according to the aligner (paired)
const unsigned int FLAG_3 = 4;      // 0x4 segment unmapped
const unsigned int FLAG_4 = 8;      // 0x8 next segment in the template unmapped
const unsigned int FLAG_5 = 16;     // 0x10 SEQ being reverse complemented
const unsigned int FLAG_6 = 32;     // 0x20 SEQ of the next segment in the template being reverse complemented
const unsigned int FORWARD_FLAG = 64;     // 0x40 the first segment in the template (forward read R1)
const unsigned int REVERSE_FLAG = 128;    // 0x80 the last segment in the template (reverse read R2)
const unsigned int FLAG_9 = 256;    // 0x100 secondary alignment
const unsigned int FLAG_10 = 512;   // 0x200 not passing filters, such as platform/vendor quality controls
const unsigned int FLAG_11 = 1024;  // 0x400 PCR or optical duplicate
const unsigned int FLAG_12 = 2048;  // 0x800 supplementary alignment


int readlength = -1;

struct al_data{
    std::string read;
    std::string contig;
    unsigned int pair_dir; 
    bool paired;
    float sequence_identity;
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // A simple combination function
        return h1 ^ h2;
    }
};

struct ContigData {
    float total_count;
    float unique_count;
    float cross_count;
    // Constructor that initializes all counts to 0.0f
    ContigData() : total_count(0.0f), unique_count(0.0f), cross_count(0.0f) {}
};

using AlnParser = std::vector<al_data>;
using AlnStat = std::pair<float, float>;
using ContigsMap = std::unordered_map<std::string, ContigData>;
using ContigsLen = std::unordered_map<std::string, int>;
using AlnMapids = std::unordered_set<std::pair<std::string, std::string>, PairHash>;
using ContigLinks = std::unordered_map<std::pair<std::string, std::string>, float, PairHash>;

void addpairlinks(
    ContigLinks& contiglinks,
    const std::string& c1,
    const std::string& c2,
    float frac_contigs_mapped) {
    
    std::string first = (c1 < c2) ? c1 : c2;
    std::string second = (c1 < c2) ? c2 : c1;
    contiglinks[std::make_pair(first, second)] += frac_contigs_mapped;
}

void construct_map(
    ContigsMap& contigs_map,
    ContigsLen& contigs_len,
    std::string line,
    unsigned int minlength) {

    float init_count = 0.0f;
    if (line.compare(0, 3, "@SQ") == 0) {
        std::string header, id, length;
        std::istringstream iss(line);
        iss >> header >> id >> length;
        
        // Extract contig id and its length
        std::string contig_id = id.substr(3);
        unsigned int contig_length = std::stoi(length.substr(3));

        // Store only if contig length is at least minlength
        if (contig_length >= minlength) {
            contigs_map.emplace(contig_id, ContigData());
            contigs_len[contig_id] = contig_length;
        }
    }
}

std::pair<int, int> get_alnpos(std::string &cigar_str) {
    int start, end, pos1, pos2 = 0;
    char op1, op2;
    // bool softclip_flag = false;

    if (sscanf(cigar_str.c_str(), "%d%c", &pos1, &op1) == 2) {
        // eg: 150M
        start = 0;
        end = pos1;
        // softclip_flag = true;
    } else {
        if (sscanf(cigar_str.c_str(), "%d%c%d%c", &pos1, &op1, &pos2, &op2) == 4) {
            // eg: 140M10S
            if(op1 == 'M') {
                start = 0;
                end = pos1;
                // if(op2 == 'S') {
                //     softclip_flag = true;
                // }
            } else {
                // eg: 10S140M
                start = pos1;
                end = start + pos2;
                // if(op1 == 'S') {
                //     softclip_flag = true;
                // }
            }
        }
    }

    if (readlength < 0) {
        readlength = start + end;
    }
    // } else {
    //     std::cout << "should not enter this part where cigar does not have perfect or soft clip\n";
    //     softclip_flag = true;
    // }

    return std::make_pair(start, end);
}


std::pair<float, float> get_seqid_alncov(std::pair<int, int> &alnpos, std::string &qual_str, std::string &md_str) {
    
    unsigned int matches = 0;
    unsigned int mismatches = 0;
    std::string mismatch_string;
    std::istringstream ss(md_str);
    unsigned int future_matches = 0;
    ss >> future_matches;

    int start = alnpos.first;
    int end = alnpos.second;

    unsigned int alignment_length = end - start;

    //  Sequence identity only considers aligned region (soft clip region is ignored)
    //  Alignment coverage is calculated w.r.t full read length
    for (unsigned int i = start; i < end; i++) {
        bool is_match;
        if (future_matches > 0) {
            future_matches--;
            is_match = true;
        } else {
            is_match = false;
            char tt;
            ss >> tt;
            int t_match;
            if(ss >> t_match) {
                future_matches = t_match;
            } else {
                //do nothing
            }
        }
        // Quality check per nt position
        // 33 is the ASCII offset
        if (qual_str[i] >= 20 + 33) {
            if(is_match){
                matches++;
            } else {
                mismatches++;
            }
        }
    }
    float seq_id, alignment_coverage;

    if (!((matches+mismatches) > 0)) {
        std::cerr << "Zero matches and mismatched position. Something wrong with the alignment\n";
        exit(1);
    }

    if (!(qual_str.length() > 0)) {
        std::cerr << "Zero quality string length. Something wrong with the alignment \n";
        exit(1);
    }

    seq_id = (static_cast<float>(matches) * 100) / (matches+mismatches);
    alignment_coverage = (static_cast<float>(alignment_length) * 100) / (qual_str.length() + start); // account for variable read length
    // std::cout << matches << " matches " << mismatches << " mismatches " <<  seq_id << " " << alignment_coverage << " " << matches+mismatches << " in alnstats \n";
    return std::make_pair(seq_id, alignment_coverage);
}

void count_match_mismatch(int& length, unsigned int& read_pos, unsigned int& mm, const std::string & qual_str) {
    for (int i = 0; i < length; ++i) {
        if (read_pos < qual_str.size() && qual_str[read_pos] >= 20 + 33) {
            mm++;
        }
    read_pos++;
    }
}

std::pair<float, float> get_seqid_alncov_strobealign(const std::string& cigar_str, const std::string& qual_str) {
    unsigned int matches = 0;
    unsigned int mismatches = 0;
    unsigned int start = 0, end = 0;
    unsigned int cigar_pos = 0;
    bool visited_aligned_region = false;

    unsigned int read_pos = 0;  // Position in the read, accounting for the start position in the CIGAR
    
    std::istringstream ss(cigar_str);
    char alnstate;
    int length;

    if (readlength < 0) {
        readlength = qual_str.length();
    }
    while (ss >> length >> alnstate) {
        switch (alnstate) {
            case 'S':
                // Always softclip occurs at the start or end of the alignment
                if (visited_aligned_region) {
                    // End soft clip means we've reached the end of the alignment
                    end = read_pos;
                } else {
                    // Start soft clip means the start of the alignment is after this
                    start = read_pos + length;
                }
                read_pos += length;
                break;
            case '=':  // Matches
                visited_aligned_region = true;
                count_match_mismatch(length, read_pos, matches, qual_str);
                break;
            case 'X':  // Mismatches
                visited_aligned_region = true;
                count_match_mismatch(length, read_pos, mismatches, qual_str);
                break;
            default:
                std::cerr << "Unknown CIGAR or not accepted operation in the alignment: " << alnstate << std::endl;
                exit(1);
        }
    }

    if (visited_aligned_region) {
        end = read_pos;
    }
    unsigned int alignment_length = end - start;

    // Calculate sequence identity and alignment coverage
    float seq_id, alignment_coverage;

    if (!(matches + mismatches > 0)) {
        std::cout << "Zero matches and mismatches. Something wrong with the alignment, skipping it!\n";
        return std::make_pair(0.0, 0.0);
    }

    if (qual_str.empty()) {
        std::cout << "Zero quality string length. Something wrong with the alignment, skipping it!\n";
        return std::make_pair(0.0, 0.0);
    }

    seq_id = (static_cast<float>(matches) * 100) / (matches + mismatches);
    alignment_coverage = (static_cast<float>(alignment_length) * 100) / (qual_str.length() + start);

    return std::make_pair(seq_id, alignment_coverage);
}

void storealnstats(AlnParser &parsedaln, AlnStat &alnstats, std::string &currentread_id, std::string &contig_id, unsigned int &pair_dir, bool & proper_pair) {
    auto ic = std::find_if(parsedaln.rbegin(), parsedaln.rend(),
                [&](const al_data& a) {return a.contig == contig_id;});
    if (ic != parsedaln.rend()){ 
        // if mate pair alignment is found
        if (ic->pair_dir != pair_dir){
            ic->sequence_identity = (ic->sequence_identity + alnstats.first) / 2.0f;
            ic->paired = true;
        } else { // update sequence identity if the same read (of the same direction) mapped to the same contig with higher sequence identity
            if (ic->sequence_identity < alnstats.first) {
                ic->sequence_identity = alnstats.first;
            }
        }
    } else { // read mapped to different contig 
        parsedaln.push_back({currentread_id, contig_id, pair_dir, proper_pair, alnstats.first});
    }
}

void counting(AlnParser &parsedaln, ContigsMap &contigs_map, ContigLinks &contig_links, float &sequenceidentity, AlnMapids& alnmapids) {
    if (parsedaln.empty()) return;
    auto max_it = std::max_element(parsedaln.begin(), parsedaln.end(),
        [](const al_data& a,const al_data& b)
        { return a.sequence_identity < b.sequence_identity;});

    // for (size_t r = 0; r < parsedaln.size(); r++) {
    //     std::cout << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln[r].sequence_identity << " complete set \n";
    //     // mapids << parsedaln[r].read << " " << parsedaln[r].contig << " " << parsedaln[r].pair_dir << " " << parsedaln[r].paired << " " << parsedaln.size() << " " << parsedaln[r].sequence_identity << "\n";
    // }

    if (max_it->sequence_identity >= sequenceidentity) {
        parsedaln.erase(
            std::remove_if(parsedaln.begin(),parsedaln.end(), [&](const al_data& a) { 
                return max_it->sequence_identity > a.sequence_identity;}),
            parsedaln.end()
            );

        if (parsedaln.size() > 0) {
            // Check if there are any elements with paired == true
            bool has_paired = std::any_of(parsedaln.begin(), parsedaln.end(), [](const al_data& a) {
                return a.paired;
            });

            // Keep only paired end read mapping (paired == true) if present and remove single read mapping
            if (has_paired) {
                parsedaln.erase(
                    std::remove_if(parsedaln.begin(), parsedaln.end(), [](const al_data& a) {
                    return !a.paired;}),
                    parsedaln.end()
                );
            }
            // Keep only the first 10 elements
            if (parsedaln.size() > 10) {
                parsedaln.resize(10);
            }

            float frac_contigs_mapped = 1.0f / parsedaln.size();

            // Update counts in contigs_map and unique_map or cross_map
            if (parsedaln.size() == 1) {
                ContigData& data = contigs_map[parsedaln[0].contig];
                data.total_count += 1.0f;
                data.unique_count += 1.0f;
                std::pair<std::string, std::string> pair = std::make_pair(parsedaln[0].read, parsedaln[0].contig);
                alnmapids.insert(pair);
                // mapids << parsedaln[0].read << " " << parsedaln[0].contig << " " << parsedaln[0].pair_dir << " " << parsedaln[0].paired << " " << parsedaln.size() << " " << parsedaln[0].sequence_identity << "\n";
            } else {    
                std::vector<std::string> contiglist;
                for (const auto &aln: parsedaln) {
                    contiglist.push_back(aln.contig);
                    ContigData& data = contigs_map[aln.contig];
                    data.total_count += frac_contigs_mapped;
                    data.cross_count += frac_contigs_mapped;
                    std::pair<std::string, std::string> pair = std::make_pair(aln.read, aln.contig);
                    alnmapids.insert(pair);
                    // mapids << aln.read << " " << aln.contig << " " << aln.pair_dir << " " << aln.paired << " " << parsedaln.size() << " " << aln.sequence_identity << "\n";
                }
                // add shared reads links to contigs
                for (std::size_t i = 0; i < contiglist.size(); ++i) {
                    for (std::size_t j = i + 1; j < contiglist.size(); ++j) {
                        addpairlinks(contig_links, contiglist[i], contiglist[j], frac_contigs_mapped);
                    }
                }
                // clear contiglist data of current read alignments
                contiglist.clear();
            }
        }
    } else {
        parsedaln.clear();
    }
}

void fractionate_countlinks(ContigLinks &contiglinks, ContigsMap &contigs_map, std::string &outdir, std::string &outputname) {
    std::ofstream linkfile;
    linkfile.open(outdir + '/' + outputname + "_countlinks");
    if (!linkfile.is_open()) {
        std::cerr << "Error opening file: " << outdir + '/' + outputname + "_countlinks" << "\n";
        return;
    }

    for (auto const & k: contiglinks) {
        const std::string &contig1 = k.first.first;
        const std::string &contig2 = k.first.second;
        float frac_links = k.second;

        auto it1 = contigs_map.find(contig1);
        auto it2 = contigs_map.find(contig2);

        if (it1 == contigs_map.end() || it2 == contigs_map.end()) {
            std::cerr << "One of the contigs (" << contig1 << ", " << contig2 << ") is not present in contigs_map. Revisit the code.\n";
            continue;
        }

        float count_1 = it1->second.total_count;
        float count_2 = it2->second.total_count;
        float normalize_factor = std::min(count_1, count_2);
        if (frac_links > normalize_factor) {
            frac_links = normalize_factor;
            std::cerr << "Error: fraction link count is greater than the minimum total counts of a pair. Revisit the code \n";
        }
        frac_links /= normalize_factor;

        // output file format: (i) pair_contig 1, (ii) pair_contig 2, 
        // (iii) fraction of value obtained from the split-mapped read, (iv) iii normalized by the total count of shortest contig in the pair
        linkfile << contig1 << " " << contig2 << " " << k.second << " " << frac_links << "\n";
    }

    linkfile.close();
}

void process_header_line(
    const std::string& line, 
    ContigsMap& contigs_map,
    ContigsLen& contigs_len,
    unsigned int minlength) {
    if ((line.rfind("@HD",0) == 0)) { // check if input alignment is unsorted
        if ((line.find("SO:unsorted") == std::string::npos) && line.find("SS") != std::string::npos) {
            std::cerr << "Input alignment file is sorted. Please provide unsorted file or ordered by read ids\n";
            exit(1);
        }
    } else { construct_map(contigs_map, contigs_len, line, minlength); };
}

void process_alignment_line(
    const std::string& line,
    ContigsMap& contigs_map,
    ContigLinks& contiglinks,
    AlnParser& parsedaln,
    float sequenceidentity,
    float read_coverage,
    AlnMapids& alnmapids,
    bool& strobealign) {
    if (contigs_map.size() == 0) {
        std::cerr << "Input sam/bam file doesn't has header. Please provide input file with header \n";
        exit(1);
    }
    std::string currentread_id, contig_id, cigar_str, qual_str, md_str, field;
    unsigned int bitflag;

    std::istringstream iss(line);
    iss >> currentread_id >> bitflag >> contig_id;

    // No reference or not of minimum length, don't process the alignment;
    if ((contig_id == "*") || (contigs_map.find(contig_id) == contigs_map.end())) return;

    // // Store read-contig pairs of all alignment lines without applying any conditions
    // std::pair<std::string, std::string> pair = std::make_pair(currentread_id, contig_id);
    // alnmapids.insert(pair);
    
    iss >> field >> field >> cigar_str;
    if (cigar_str.find_first_of("IDNPH*") != std::string::npos) {
        // indel, hard clipping, don't process the alignment;
        if (bitflag & PAIRED_FLAG && !parsedaln.empty()) {
            parsedaln.erase(std::remove_if(parsedaln.begin(), parsedaln.end(),
            [contig_id](const al_data& element) {
                return element.contig == contig_id;
            }),
            parsedaln.end());
        }
        return; 
    }

    iss >> field >> field >> field >> field >> qual_str; // >> field >> field;

    std::pair<float, float> alnstats;
    if (!strobealign) {
        try {
            // *** old code ***
            // if (field.find("XS") != std::string::npos) {
            //     // suboptimal alignment score is given;
            //     iss >> field >> field >> field >> field >> field >> md_str;
            // } else {
            //     iss >> field >> field >> field >> field >> md_str;
            // }
            // md_str = md_str.substr(5, md_str.length());
            // *** old code ***

            // This should work for both bowtie2 and bwa mem
            while (iss >> field) {
                if (field.find("MD:Z:") != std::string::npos) {
                    md_str = field.substr(5, field.length());
                    break;
                }
            }
            if (md_str.empty()) {
                std::cerr << "MD field not found in the SAM alignment\n";
                exit(1);
            }

        } catch (const std::exception& e) {
            std::cerr << "Error: Input sam file should be in bowtie2 format\n";
            std::cerr << "This tool supports bowtie2 (default), bwa mem or strobealign format only!\n";
            std::cerr << "Use --strobealign flag if alignment was generated using strobealign\n";
            exit(1);
        }
        auto alnpos = get_alnpos(cigar_str);

        // clip at both ends, don't process the alignment;
        if (std::get<1>(alnpos) == 0) return;

        alnstats = get_seqid_alncov(alnpos, qual_str, md_str);
    } else {
        alnstats = get_seqid_alncov_strobealign(cigar_str, qual_str);
    }

    if ((!std::isnan(alnstats.first)) && (alnstats.second >= read_coverage)) {
        // get read direction
        // TODO: check this is required for single end
        unsigned int pair_dir = (bitflag & FORWARD_FLAG) ? 1: (bitflag & REVERSE_FLAG) ? 2: 0;
        if (pair_dir == 0){
            std::cerr << "input reads are not paired\n";
            exit(1);
        }
        
        // get if alignment is paired
        bool proper_pair = bitflag & PAIRED_FLAG;

        if (parsedaln.empty() || parsedaln.rbegin()->read != currentread_id) {
            // new read alignment
            if (!parsedaln.empty()) {
                counting(parsedaln, contigs_map, contiglinks, sequenceidentity, alnmapids);
                parsedaln.clear();
            }
            // store new read alignment
            parsedaln.push_back({currentread_id, contig_id, pair_dir, proper_pair, alnstats.first});
        } else { // continue store alignment of current read
            storealnstats(parsedaln, alnstats, currentread_id, contig_id, pair_dir, proper_pair);
        }
    }

}

void write_counts(
    const std::string& outdir,
    const std::string& outputname,
    const ContigsMap& contigs_map,
    const ContigsLen& contigs_len,
    bool coverage) {
    std::ofstream outfile(outdir + '/' + outputname + "_count");
    std::ofstream uniquefile(outdir + '/' + outputname + "_uniqcount");
    std::ofstream crossfile(outdir + '/' + outputname + "_crosscount");
    std::ofstream coveragefile;
    if (coverage){
        coveragefile.open(outdir + '/' + outputname + "_coverage");
    }
    float contig_cov;
    for (const auto& k : contigs_map) {
        const ContigData& data = k.second;
        outfile << k.first << " " << outputname << " " << data.total_count << "\n";
        uniquefile << k.first << " " << outputname << " " << data.unique_count << "\n";
        crossfile << k.first << " " << outputname << " " << data.cross_count << "\n";
        if (coverage) {
            contig_cov =  (data.total_count * readlength) / contigs_len.at(k.first);
            coveragefile << k.first << " " << outputname << " " << contig_cov << "\n";
        }
    }
    outfile.close();
    crossfile.close();
    uniquefile.close();
    if (coverage) {
        coveragefile.close();
    }
}

// TO DO LIST
// Flags in sam files are considered only as per bowtie2 output. Need to generalize or check compatibility for bwa-mem and minimap2 output
// 

int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    // never use scanf or any library which would use scanf, but it makes input much faster
    std::ios::sync_with_stdio(false);

    // Mandatory arguments
    std::string outdir;
    std::string outputname;
    
    // Optional arguments with default values
    unsigned int minlength = 1000;
    bool coverage = true;
    bool paired = true;
    bool onlymapids = false;
    bool strobealign = false;
    float qcov = 99.0f;
    float seq_id = 97.0f;

    // Structure for long options
    static struct option long_options[] = {
        {"no-coverage", no_argument, 0, 'c'},
        {"single", no_argument, 0, 'p'},
        {"only-mapids", no_argument, 0, 'm'},
        {"strobealign", no_argument, 0, 'r'},
        {"qcov", required_argument, 0, 'q'},
        {"seq-id", required_argument, 0, 's'},
        {"minlength", required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };
    
    if (argc < 3 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        std::cerr << "Usage: aligner command | "<< argv[0] << " outdir outputname [--minlength N] [--no-coverage] [--single] [--only-mapids] [--qcov X] [--seq-id Y] [--strobealign]\n";
        std::cerr << "Options:\n";
        std::cerr << "  <outdir>            Directory where output files will be stored.\n";
        std::cerr << "  <outputname>        Base name for the output files (without extensions).\n\n";
        std::cerr << "  --minlength N       Minimum length of contigs to be considered. (Default: N=1000)\n";
        std::cerr << "                      Alignments for contigs shorter than this length will be ignored.\n\n";
        std::cerr << "  --no-coverage       Do not output coverage. (Optional)\n";
        std::cerr << "                      This flag disables the output of contig coverage.\n\n";
        std::cerr << "  --single            Process input as single-end reads. (Optional)\n";
        std::cerr << "                      By default, paired-end reads are expected unless this flag is set.\n\n";
        std::cerr << "  --strobealign       Alignment is generated from stobealign with `--eqx` flag. (Optional, default=bowtie2)\n\n";
        std::cerr << "  --only-mapids       Output only the mapped pairs of read and contig identifiers. (Optional)\n";
        std::cerr << "                      This flag restricts the output to just the IDs of reads and mapped contigs.\n\n";
        std::cerr << "  --qcov X            Minimum query/read coverage threshold X. (Optional, default=99.0%)\n";
        std::cerr << "                      Specifies the minimum percentage of query sequence that must be aligned.\n\n";
        std::cerr << "  --seq-id Y          Minimum sequence identity percentage Y. (Optional, default=97.0%)\n";
        std::cerr << "                      Filters alignments by requiring at least Y% identity.\n\n";
        std::cerr << "  -h, --help          Display this help message and exit.\n";
        return 1;
    }

    outdir = argv[1];
    outputname = argv[2];
    // unsigned int minlength = (argc > 3) ? std::stoul(argv[3]) : 1000;
    // Parse optional arguments
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "cpmrq:s:l:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'c':
                coverage = false;
                break;
            case 'p':
                paired = false;
                break;
            case 'm':
                onlymapids = true;
                break;
            case 'r':
                strobealign = true;
                break;
            case 'q':
                qcov = std::stof(optarg);
                break;
            case 's':
                seq_id = std::stof(optarg);
                break;
            case 'l':
                minlength = std::stoi(optarg);
                break;
            case '?':
                // getopt_long prints an error message automatically
                return 1;
            default:
                break;
        }
    }

    ContigsMap contigs_map;
    ContigsLen contigs_len;
    ContigsMap unique_map;
    ContigsMap cross_map;
    ContigLinks contiglinks;

    AlnMapids alnmapids;

    AlnParser parsedaln;
    float sequenceidentity = seq_id;
    float read_coverage = qcov;
    std::ofstream samfile;
    std::ofstream mapids(outdir + '/' + outputname + "_mapids");
    if (!onlymapids) {
        samfile.open(outdir + '/' + outputname + ".sam",  std::ios::binary);
        std::cout << "Output name:" << outputname << "\n";
        std::cout << "Minimum length:" << minlength << "\n";
        std::cout << "Coverage: " << coverage << "\n";
        std::cout << "Paired: " << paired << "\n";
        std::cout << "Query coverage: " << qcov << "%\n";
        std::cout << "Sequence identity: " << seq_id << "%\n";
    }
    std::string line;
    while(std::getline(std::cin, line)) {
        // write all line before applying any conditions and proceed with continue
        if (!onlymapids) {
            samfile.write(line.c_str(), line.size());
            samfile.put('\n');
        }
        if (line.empty()) continue;

        if (line[0] == '@') {
            process_header_line(line, contigs_map, contigs_len, minlength);
        } else {
            process_alignment_line(line, contigs_map, contiglinks, parsedaln, sequenceidentity, read_coverage, alnmapids, strobealign);
        }
    }
    if (readlength < 0) {
        std::cerr << "Error: readlength is not computed\n";
        exit(1);
    }
    // while end
    // process last read alignment(s)
    counting(parsedaln, contigs_map, contiglinks, sequenceidentity, alnmapids);
    if (!onlymapids) {
        fractionate_countlinks(contiglinks, contigs_map, outdir, outputname);
        // multiple read length by 2 if paired-end reads
        if (paired) {
            readlength = readlength * 2;
        }
        write_counts(outdir, outputname, contigs_map, contigs_len, coverage);
        samfile.close();
    }

    for (const auto& pair : alnmapids) {
        mapids << pair.first << " " << pair.second << std::endl;
    }
    parsedaln.clear();
    mapids.close();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << "Alignment file for sample " << outputname << " processed in " << duration.count() << " seconds\n";

    return EXIT_SUCCESS;
}
