#include <iostream>
#include <thread>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <random>
#include <mutex>
#include <functional>
#include <string>
#include <cstring> 



struct Task {
    std::string type;
    double argument;
    int iter;
    int argument2 = 0;
};

template<typename T>
class Server {
public:
    Server() : running(true) {}

    void start() {
        thread_ = std::thread(&Server::run, this);
    }

    void stop() {
        running = false;
        thread_.join();
    }

    size_t add_task(Task task) {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_[task_id_] = task;
        return task_id_++;
    }

    T request_result(size_t id) {
        while (true) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (results_.find(id) != results_.end()) {
                T result = results_[id];
                results_.erase(id);
                return result;
            }
        }
    }

private:
    void run() {
        while (running) {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& task : tasks_) {
                T result;
                if(task.second.type == "Sinus"){
                    result = std::sin(task.second.argument);
                }
                else if(task.second.type == "SquareRoot"){
                    result = std::sqrt(task.second.argument);
                }
                else{
                    result = std::pow(task.second.argument, task.second.argument2);
                }
                results_[task.first] = result;
            }
            tasks_.clear();
        }
    }

private:
    std::thread thread_;
    std::unordered_map<size_t, Task> tasks_;
    std::unordered_map<size_t, T> results_;
    size_t task_id_ = 0;
    std::mutex mutex_;
    bool running;
};

void client(Server<double>& server, Task task, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    std::uniform_int_distribution<int> dist2(0, 10);

    for (size_t i = 0; i < task.iter; ++i) {
        double argument = dist(gen); // Генерируем случайный аргумент
        int argument2 = 0;
        if (task.type == "Power"){
            argument2 = dist2(gen); // Генерируем случайное значение для второго аргумента в случае Power
        }

        Task current_task = {task.type, argument, task.iter, argument2};
        size_t task_id = server.add_task(current_task);
        double result = server.request_result(task_id);

        file << "Task " << task_id << ": arg1 = " << argument;
        if (task.type == "Power") {
            file << " arg2 = " << argument2;
        }
        file << " Result = " << result << std::endl;
    }

    file.close();
}


int main(int argc, char *argv[]) {

    Task client1, client2, client3;

    if((std::strcmp(argv[1], "Sinus") != 0 && std::strcmp(argv[1], "SquareRoot") != 0 && std::strcmp(argv[1], "Power") != 0)||
        (std::strcmp(argv[3], "Sinus") != 0 && std::strcmp(argv[3], "SquareRoot") != 0 && std::strcmp(argv[3], "Power") != 0)||
        (std::strcmp(argv[5], "Sinus") != 0 && std::strcmp(argv[5], "SquareRoot") != 0 && std::strcmp(argv[5], "Power") != 0)||
        argc != 7 || std::strcmp(argv[1], "help") == 0){
        std::cerr << "Введите аргументы в формате: функция (Sinus, SquareRoot, Power) и количество операций (от 5 до 10000) для каждого клиента," << '\n'
         <<"последовательно, например: Sinus 20 SquareRoot 30 Power 25";
    }
    else{
        client1.type = argv[1];
        client1.iter = atof(argv[2]);
        client2.type = argv[3];
        client2.iter = atof(argv[4]);
        client3.type = argv[5];
        client3.iter = atof(argv[6]);
    }
    Server<double> server;
    server.start();

    std::thread client_serv1(client, std::ref(server), client1, "Sinus_results.txt");
    std::thread client_serv2(client, std::ref(server), client2, "sqrt_results.txt");
    std::thread client_serv3(client, std::ref(server), client3, "power_results.txt");

    client_serv1.join();
    client_serv2.join();
    client_serv3.join();

    server.stop();

    return 0;
}