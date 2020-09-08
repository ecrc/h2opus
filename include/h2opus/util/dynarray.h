#ifndef __H2OPUS_DYN_ARRAY_H__
#define __H2OPUS_DYN_ARRAY_H__

template <class T> class DynamicArray
{
  public:
    DynamicArray()
    {
        m_size = 0;
        m_data = NULL;
    }

    ~DynamicArray()
    {
        if (m_data)
            delete[] m_data;
    };

    size_t size() const
    {
        return m_size;
    }

    T &operator[](size_t index) const
    {
        assert(index < m_size);
        return m_data[index];
    }

    void resize(size_t new_size)
    {
        if (m_size != new_size)
        {
            delete[] m_data;
            m_size = new_size;
            m_data = new T[m_size];
        }
    }

  private:
    size_t m_size;
    T *m_data;
};

#endif
