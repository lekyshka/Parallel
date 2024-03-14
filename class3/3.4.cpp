#include <future>
#include <iostream>
#include <thread>
#include <utility>

void product(std::promise<int> &&intPromise, int a, int b)
{
    intPromise.set_value(a * b);
}

struct Div
{
    void operator()(std::promise<int> &&intPromise, int a, int b) const
    {
        try
        {   
            if (b == 0) throw std::runtime_error("Div zero!");
            int temp = a / b;
            intPromise.set_value(temp);
        }
        catch (...)
        {
            try
            {
                // store anything thrown in the promise
                intPromise.set_exception(std::current_exception());
            }
            catch (...) // set_exception() may throw too
            {
            }
        }
    }
};

int main()
{
    int a = 20;
    int b = 0;

    // define the promises
    std::promise<int> prodPromise;
    std::promise<int> divPromise;

    // get the futures
    std::future<int> prodResult = prodPromise.get_future();
    std::future<int> divResult = divPromise.get_future();

    try
    {
        // calculate the result in a separate thread
        std::jthread prodThread(product, std::move(prodPromise), a, b);
        Div div;
        std::jthread divThread(div, std::move(divPromise), a, b);

        // get the result

        std::cout << a <<  " * " << b << " = " << prodResult.get() << std::endl;
        std::cout << a <<  " / " << b << " = " << divResult.get() << std::endl;
    }
    catch (const std::exception &excep)
    {
        std::cout << "\nException from the thread: " << excep.what() << '\n';
    }
    
    std::cout << std::endl;
}