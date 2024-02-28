#include <iostream>
#include <thread>
#include <string>
#include <functional>

void helloFun(std::string &str)
{
    std::cout << "Hello from a function, " + str << std::endl;
}

class HelloThread
{
public:
    HelloThread(std::string arg0): _str(arg0){}

    void operator()() const
    {
        std::cout << "Hello from a functor, " + this->_str << std::endl;
    }
private:
    std::string _str;
};

int main()
{
    std::string thread1_arg{"plus argument 1"};
    std::string thread2_arg{"plus argument 2"};
    std::string thread3_arg{"plus argument 3"};

    std::thread t1(helloFun, std::ref(thread1_arg));

    HelloThread hello_class(thread2_arg);
    std::thread t2(hello_class);

    std::thread t3([thread3_arg]{std::cout << "Hello from a lambda, " 
                    + thread3_arg << std::endl;
                    });

    t1.join();
    t2.join();
    t3.join();

    std::cout << std::endl;
    return 0;
}