#include <iostream>
#include <string>
#include <thread>
#include <chrono>

void thread_fn() {
	using namespace std::chrono_literals;

  	std::this_thread::sleep_for (1s);
  	std::cout << "One\n";
  	std::this_thread::sleep_for (1s);
  	std::cout << "Two\n";  
}

int main()
{
    std::jthread t1(thread_fn);
  
    return 0; 
}