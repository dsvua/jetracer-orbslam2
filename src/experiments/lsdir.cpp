#include <string>
#include <iostream>
#include <glob.h>
#include <vector>
using std::vector;
using namespace std;

vector<string> globVector(const string& pattern, int flags){
    glob_t glob_result;
    glob(pattern.c_str(),flags,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

int main(void)
{
    vector<string> dirs = globVector((string)"./*", GLOB_TILDE | GLOB_ONLYDIR);
    vector<string> files;
    for(auto&& x: dirs){
        std::cout << x << std::endl;
    }

    for(auto&& dir: dirs){
        for(auto&& file: globVector((string)(dir + "/*"), GLOB_TILDE)){
            std::cout << file << std::endl;
            files.push_back(file);
        }
    }

    for(auto&& file: files){
        std::cout << file << std::endl;
    }

}