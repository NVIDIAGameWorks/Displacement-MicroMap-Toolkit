#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <Python.h>

#include <nvmath/nvmath.h>

#include <meshops/meshops_operations.h>
#include <meshops/meshops_mesh_view.h>

namespace py = pybind11;
using namespace meshops;

template<int Vdim, typename T, typename V>
bool numpyArrayToVector(const py::array_t<T, py::array::c_style | py::array::forcecast>& array, std::vector<V>& vector)
{
    if (array.size() == 0)
    {
        return true;
    }

    auto arrayProperties = array.request();

    unsigned int nDim0 = (unsigned int)arrayProperties.shape[0];

    if(arrayProperties.size * arrayProperties.itemsize != static_cast<decltype(arrayProperties.itemsize)>(nDim0 * sizeof(V)))
    {
      throw std::runtime_error("input array shape not compatible with vector");
    }

    vector.resize(nDim0);

    V *values = reinterpret_cast<V*>(arrayProperties.ptr);
    
    memcpy(&vector[0], values, sizeof(V) * nDim0);

#if 0
    for (int i = 0; i < vector.size(); ++i)
    {
        std::cout << "(" << vector[i].x << ", " << vector[i].y << ", " << vector[i].z << ")" << std::endl;
    }
#endif

    return true;
}

template<int Vdim, typename V, typename T>
bool vectorToNumpyArray(const std::vector<V>& vector, py::array_t<T>& array)
{
    if (vector.size() == 0)
    {
        array = py::array_t<T>();
        return true;
    }

    array = py::array_t<T>(Vdim * vector.size(), (T*)vector.data());

    if(Vdim > 1)
    {
      std::vector<py::size_t> shape({vector.size(), Vdim});
      array.resize(py::array::ShapeContainer(shape));
    }

#if 0
    auto arrayProperties = array.request();

    T * arr = static_cast<T *>(arrayProperties.ptr);
    for (int i = 0; i < Vdim * vector.size(); ++i)
    {
        std::cout << "(" << arr[i] << ")" << std::endl;
    }
#endif

    return true;
}

template<int Vdim, typename T, typename V>
bool numpyArrayToArrayView(const py::array_t<T, py::array::c_style | py::array::forcecast>& array, meshops::ArrayView<V>& vector)
{
    if(array.is_none())
    {
        return true;
    }

    if(array.size() == 0 || array.shape() == 0)
    {
        return true;
    }

    auto arrayProperties = array.request();

    unsigned int nDim0 = (unsigned int)arrayProperties.shape[0];
    
    if (nDim0 == 0)
    {
        return false;
    }

    V *values = reinterpret_cast<V*>(arrayProperties.ptr);

    if(arrayProperties.size * arrayProperties.itemsize != static_cast<decltype(arrayProperties.itemsize)>(static_cast<size_t>(nDim0 * sizeof(V))))
    {
        throw std::runtime_error("input array shape not compatible with array view");
    }

    vector = meshops::ArrayView<V>(values, nDim0, sizeof(V));

#if 0
    for (int i = 0; i < vector.size(); ++i)
    {
        std::cout << "(" << vector[i].x << ", " << vector[i].y << ", " << vector[i].z << ")" << std::endl;
    }
#endif

    return true;
}

template<int Vdim, typename V, typename T>
bool arrayViewToNumpyArray(const meshops::ArrayView<V>& vector, py::array_t<T>& array)
{
    if (vector.size() == 0)
    {
        array = py::array_t<T>();
        return true;
    }

    array = py::array_t<T>(Vdim * vector.size(), (T*)vector.data());
    
    if(Vdim > 1)
    {
      std::vector<py::size_t> shape({vector.size(), Vdim});
      array.resize(py::array::ShapeContainer(shape));
    }

#if 0
    auto arrayProperties = array.request();

    T * arr = static_cast<T *>(arrayProperties.ptr);
    for (int i = 0; i < Vdim * vector.size(); ++i)
    {
        std::cout << "(" << arr[i] << ")" << std::endl;
    }
#endif

    return true;
}

bool numpyArrayToMatrix(py::array_t<float>& array, micromesh::Matrix_float_4x4& matrix)
{
    auto arrayProperties = array.request();

    if(arrayProperties.ndim != 2)
    {
        throw std::runtime_error("input array shape does not have two dimensions");
    }

    unsigned int nDim0 = (unsigned int)arrayProperties.shape[0];
    unsigned int nDim1 = (unsigned int)arrayProperties.shape[1];

    if (nDim0 != 4 || nDim1 != 4)
    {
        throw std::runtime_error("input array shape not compatible with matrix");
    }

    float *values = reinterpret_cast<float*>(arrayProperties.ptr);

    for (int iColumn = 0; iColumn < 4; ++iColumn)
    {
        matrix.columns[iColumn] = micromesh::Vector_float_4{values[0], values[1], values[2], values[3]};
        values += 4;
    }

    return true;
}
