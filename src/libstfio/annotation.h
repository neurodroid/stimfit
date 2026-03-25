#pragma once
//! Describes the attributes of an annotation.
#include <cstddef>

#include "./core.h"

class StfioDll Annotation {
public:
    //! Constructor
    explicit Annotation(std::size_t onset, size_t duration);
    //! Destructor
    ~Annotation();

    std::size_t GetAnnotationPosition();
    std::size_t GetAnnotationDuration();
private:
    std::size_t position;
    std::size_t duration;
};
