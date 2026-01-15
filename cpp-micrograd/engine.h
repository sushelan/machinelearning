#ifndef ENGINE_H
#define ENGINE_H

#include <memory>
#include <string>
#include <unordered_set>
#include <functional>

class Value
{

private:
    float data;
    float grad;
    // use shared ptr since multiple nodes can reference same parent
    // nodes can survive at runtime
    std::unordered_set<std::shared_ptr<Value>> _prev;
    std::function<void()> _backward;
    std::string _op;

public:
    Value(float data, std::unordered_set<std::shared_ptr<Value>> _prev = {}, std::string _op = "");

    void set_data(float data);
    void set_grad(float grad);
    float get_data() const;
    float get_grad() const;
    std::unordered_set<std::shared_ptr<Value>> get_prev() const;

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other);

    // really only needed for tanh and sigmoid
    std::shared_ptr<Value> pow(const std::shared_ptr<Value> &other);
    void backward();
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> pow(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

#endif