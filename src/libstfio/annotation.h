#pragma once
//! Describes the attributes of an annotation.
#include <cstddef>
class Annotation {
public:
    //! Constructor
    explicit Annotation(std::size_t onset, size_t duration);
    //! Destructor
    ~Annotation();

    std::size_t GetAnnotationPosition();
private:
    std::size_t position;
    std::size_t duration;
};
