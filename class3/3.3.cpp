#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>

std::mutex mut;

void workOnResource()
{
    mut.lock();
    std::cout << "lock \n";
    std::cout << "id thread " << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    mut.unlock();
}

int main()
{
    std::vector<std::thread> threads;
    size_t num_threads = 10;

    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread(workOnResource));
    }

    for (size_t i = 0; i < num_threads; ++i)
    {
        threads[i].join();
    }
    
    return 0;
}

