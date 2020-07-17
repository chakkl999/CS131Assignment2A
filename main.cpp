#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int inputImage[1000][1000];
int outputImage[1000][1000];
int numChunks;
int chunkSize;
std::map<int,std::vector<int>> threadInfo;

/* ****************Change and add functions below ***************** */
void compute_prewitt_static()
{
    int chunkSize = static_cast<int>(ceil(static_cast<double>(image_height)/numChunks));
    int tid;
    #pragma omp parallel private(tid) shared(inputImage, outputImage, image_width, image_height, chunkSize)
    {
        int maskX[3][3] = {
            {1,0,-1},
            {1,0,-1},
            {1,0,-1}
        };
        int maskY[3][3] = {
            {1,1,1},
            {0,0,0},
            {-1,-1,-1}
        };
        tid = omp_get_thread_num();
        #pragma omp for schedule(static, chunkSize)
        for(int x = 0; x < image_height; ++x){
            threadInfo[tid].emplace_back(x);
            for(int y = 0; y < image_width; ++y){
                int grad_x = 0, grad_y = 0, grad = 0;
                if(x == 0 || x ==(image_height-1) || y == 0 || y == (image_width-1))
                    grad = 0;
                else {
                    for(int i = -1; i<= 1; ++i) {
                        for(int j = -1; j <= 1; ++j) {
                            grad_x += (inputImage[x+i][y+j] * maskX[i+1][j+1]);
                            grad_y += (inputImage[x+i][y+j] * maskY[i+1][j+1]);
                        }
                    }
                    grad = static_cast<int>(sqrt((grad_x * grad_x) + (grad_y * grad_y)));
                }
                if(grad < 0)
                    grad = 0;
                else if(grad > 255)
                    grad = 255;
                outputImage[x][y] = grad;
            }
        }
    }
}

void compute_prewitt_dynamic()
{
    int chunkSize = static_cast<int>(ceil(static_cast<double>(image_height)/numChunks));
    int tid;
    #pragma omp parallel private(tid) shared(inputImage, outputImage, image_width, image_height, chunkSize)
    {
        int maskX[3][3] = {
            {1,0,-1},
            {1,0,-1},
            {1,0,-1}
        };
        int maskY[3][3] = {
            {1,1,1},
            {0,0,0},
            {-1,-1,-1}
        };
        tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, chunkSize)
        for(int x = 0; x < image_height; ++x){
            threadInfo[tid].emplace_back(x);
            for(int y = 0; y < image_width; ++y){
                int grad_x = 0, grad_y = 0, grad = 0;
                if(x == 0 || x ==(image_height-1) || y == 0 || y == (image_width-1))
                    grad = 0;
                else {
                    for(int i = -1; i<= 1; ++i) {
                        for(int j = -1; j <= 1; ++j) {
                            grad_x += (inputImage[x+i][y+j] * maskX[i+1][j+1]);
                            grad_y += (inputImage[x+i][y+j] * maskY[i+1][j+1]);
                        }
                    }
                    grad = static_cast<int>(sqrt((grad_x * grad_x) + (grad_y * grad_y)));
                }
                if(grad < 0)
                    grad = 0;
                else if(grad > 255)
                    grad = 255;
                outputImage[x][y] = grad;
            }
        }
    }
}

/* **************** Change the function below if you need to ***************** */

int main(int argc, char* argv[])
{
    if(argc != 5)
    {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename> <# of chunks> <a1/a2>" << std::endl;
        return 0;
    }

    std::ifstream file(argv[1]);
    if(!file.is_open())
    {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    numChunks  = std::atoi(argv[3]);

    std::cout << "Detect edges in " << argv[1] << " using OpenMP threads with\n" << std::endl;

    /* ******Reading image into 2-D array below******** */

    std::string workString;
    /* Remove comments '#' and check image format */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            if( workString.at(1) != '2' ){
                std::cout << "Input image is not a valid PGM image" << std::endl;
                return 0;
            } else {
                break;
            }
        } else {
            continue;
        }
    }
    /* Check image size */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        } else {
            continue;
        }
    }

    /* Check image max shades */
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        } else {
            continue;
        }
    }
    /* Fill input image matrix */
    int pixel_val;
    for( int i = 0; i < image_height; i++ )
    {
        if( std::getline(file,workString) && workString.at(0) != '#' ){
            std::stringstream stream(workString);
            for( int j = 0; j < image_width; j++ ){
                if( !stream )
                    break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        } else {
            continue;
        }
    }

    /************ Call functions to process image *********/
    std::string opt = argv[4];
    chunkSize = static_cast<int>(ceil(static_cast<double>(image_height)/numChunks));
    if( !opt.compare("a1") )
    {
//        std::cout << "Static" << std::endl;
        double dtime_static = omp_get_wtime();
        compute_prewitt_static();
        dtime_static = omp_get_wtime() - dtime_static;
        std::cout << "Took " << dtime_static << " second(s).\n" << std::endl;
    } else {
        double dtime_dyn = omp_get_wtime();
        compute_prewitt_dynamic();
        dtime_dyn = omp_get_wtime() - dtime_dyn;
        std::cout << "Took " << dtime_dyn << " second(s).\n" << std::endl;
    }

    /* ********Start writing output to your file************ */
    std::ofstream ofile(argv[2]);
    if( ofile.is_open() )
    {
        ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
        for( int i = 0; i < image_height; i++ )
        {
            for( int j = 0; j < image_width; j++ ){
                ofile << outputImage[i][j] << " ";
            }
            ofile << "\n";
        }
    } else {
        std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
        return 0;
    }

    for(std::map<int, std::vector<int>>::iterator it = threadInfo.begin(); it != threadInfo.end(); ++it)
    {
        for(int i = 0; i < it->second.size(); i += chunkSize)
            std::cout << "Thread " << it->first << " -> Processing Chunk starting at Row " << it->second[i] << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
