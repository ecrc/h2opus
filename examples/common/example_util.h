#ifndef __EXAMPLE_UTIL_H__
#define __EXAMPLE_UTIL_H__

#include <map>
#include <cstdio>
#include <string>

// Simple command line argument parser
struct H2OpusArgParser
{
  private:
    typedef std::map<std::string, std::string> ArgMap;

    struct Option
    {
        std::string long_name, short_name, description;
        std::string default_value;

        Option(const char *short_name, const char *long_name, const char *description)
        {
            this->short_name = std::string(short_name);
            this->long_name = std::string(long_name);
            this->description = std::string(description);
        }
    };

    ArgMap arg_map;
    bool validArgs;
    std::vector<Option> options;

    ArgMap::iterator findOption(Option &option)
    {
        ArgMap::iterator option_it = arg_map.find(option.short_name);
        if (option_it == arg_map.end())
            option_it = arg_map.find(option.long_name);
        return option_it;
    }

    template <typename T> T parseOption(std::string &value);

    template <typename T> std::string toString(const T &t)
    {
        return std::to_string(t);
    }

    std::string toString(const char *t)
    {
        return t;
    }

    std::string toString(const std::string &t)
    {
        return t;
    }

  public:
    H2OpusArgParser()
    {
        validArgs = true;
    }

    void setArgs(int argc, char **argv)
    {
        std::pair<ArgMap::iterator, bool> insert_ret;
        bool adding_arg = false;

        for (int i = 1; i < argc && validArgs; i++)
        {
            int insert_pos = 0;

            if (argv[i][0] == '-')
            {
                if (strlen(argv[i]) > 1 && argv[i][1] == '-')
                    insert_pos = 2;
                else
                    insert_pos = 1;

                insert_ret = arg_map.insert(
                    std::pair<std::string, std::string>(std::string(argv[i] + insert_pos), std::string("")));
                if (insert_ret.second == false)
                {
                    fprintf(stderr, "Argument %d %s was repeated\n", i, argv[i] + insert_pos);
                    validArgs = false;
                }
                adding_arg = true;
            }
            else if (adding_arg)
            {
                insert_ret.first->second = std::string(argv[i]);
                adding_arg = false;
            }
            else
            {
                fprintf(stderr, "Argument %d %s is invalid\n", i, argv[i]);
                validArgs = false;
            }
        }
    }

    // Calling this adds to the list of options - user is expected to call it once per option
    template <typename T>
    T option(const char *short_name, const char *long_name, const char *description, T default_value)
    {
        Option opt(short_name, long_name, description);
        opt.default_value = toString(default_value);

        options.push_back(opt);
        ArgMap::iterator option_it = findOption(opt);
        if (option_it == arg_map.end())
            return default_value;

        return parseOption<T>(option_it->second);
    }

    bool flag(const char *short_name, const char *long_name, const char *description, bool default_value)
    {
        Option opt(short_name, long_name, description);
        opt.default_value = (default_value ? "true" : "false");

        options.push_back(opt);
        ArgMap::iterator option_it = findOption(opt);
        if (option_it == arg_map.end())
            return default_value;

        return true;
    }

    bool valid()
    {
        return validArgs;
    }

    void printUsage()
    {
        std::cout << "Accepted arguments:\n";
        for (int i = 0; i < (int)options.size(); i++)
            fprintf(stdout, "\t-%-4s, --%-20s \t %s (Default value: %s)\n", options[i].short_name.c_str(),
                    options[i].long_name.c_str(), options[i].description.c_str(), options[i].default_value.c_str());
    }
};

template <> int H2OpusArgParser::parseOption(std::string &value)
{
    try
    {
        int i = std::stoi(value);
        return i;
    }
    catch (std::invalid_argument const &e)
    {
        fprintf(stderr, "Bad input: std::invalid_argument thrown: %s\n", e.what());
    }
    catch (std::out_of_range const &e)
    {
        fprintf(stderr, "Integer overflow: std::out_of_range thrown: %s\n", e.what());
    }

    validArgs = false;
    return 0;
}

template <> double H2OpusArgParser::parseOption(std::string &value)
{
    try
    {
        double i = std::stod(value);
        return i;
    }
    catch (std::invalid_argument const &e)
    {
        fprintf(stderr, "Bad input: std::invalid_argument thrown: %s\n", e.what());
    }
    catch (std::out_of_range const &e)
    {
        fprintf(stderr, "Integer overflow: std::out_of_range thrown: %s\n", e.what());
    }

    validArgs = false;
    return 0;
}

template <> float H2OpusArgParser::parseOption(std::string &value)
{
    try
    {
        float i = std::stof(value);
        return i;
    }
    catch (std::invalid_argument const &e)
    {
        fprintf(stderr, "Bad input: std::invalid_argument thrown: %s\n", e.what());
    }
    catch (std::out_of_range const &e)
    {
        fprintf(stderr, "Integer overflow: std::out_of_range thrown: %s\n", e.what());
    }

    validArgs = false;
    return 0;
}

template <> const char *H2OpusArgParser::parseOption(std::string &value)
{
    return value.c_str();
}

inline H2Opus_Real vec_diff(H2Opus_Real *x1, H2Opus_Real *x2, int n)
{
    H2Opus_Real norm_x = 0, diff = 0;
    for (int i = 0; i < n; i++)
    {
        H2Opus_Real entry_diff = x1[i] - x2[i];

        diff += entry_diff * entry_diff;
        norm_x += x1[i] * x1[i];
    }
    return sqrt(diff / norm_x);
}

#ifdef H2OPUS_USE_MPI
// Convenience routine to call MPI initialization
#include <h2opus/distributed/comm_wrapper.h>
void initMPI(int argc, char **argv, bool thr)
{
    if (!thr)
    {
        mpiErrchk(MPI_Init(&argc, &argv));
    }
    else
    { /* Use MPI_THREAD_MULTIPLE if you want to use threaded MPI calls within H2OPUS */
        int provided_support;
        mpiErrchk(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_support));
    }
}
#endif

#endif
