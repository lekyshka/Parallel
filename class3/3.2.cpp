#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    std::jthread non_interrupt([]
                               {
        int counter{0};
        while (counter < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cerr << "nonInterruptable: " << counter << '\n';
            ++counter;
        } });

    std::jthread interrupt([](std::stop_token stoken)
                           {
        int counter{0};
        while (counter < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (stoken.stop_requested()) return;
            std::cerr << "interruptable: " << counter << '\n';
            ++counter;
        } });

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cerr << '\n';
    std::cerr << "Main thread interrupts both jthreads"
              << "\n";
    non_interrupt.request_stop();
    interrupt.request_stop();

    std::cout << '\n';
}

// struct S
// {
//     char a;     // область 1
//     int b : 5;  // область 2
//     int c : 11, // область 2 (продолжение)
//         : 0,
//         d : 8;     // область 3
//     int e;         // область 4
//     double f;      // область 5
//     std::string g; // несколько областей
// } obj;