#include "Annotation.h"

Annotation::Annotation(std::size_t onset, size_t duration) :
    position(onset), duration(duration)
{
}

Annotation::~Annotation()
{
}

std::size_t Annotation::GetAnnotationPosition()
{
    return this->position;
}
