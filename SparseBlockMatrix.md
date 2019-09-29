## Sparse Block Matrices

These block matrices were made to allow for dense matrix operations with fixed size sub matrices.

### Sparse Block Matrix
```cpp
class SparseBlockMatrix<ScalarType, VariableGroup<V1, V2, ...>> {
  
  // Gets a block matrix at a given index.
  MatrixBlock<ScalarType, T1::Dimension, T2::Dimension> get(TypedIndex<T1> row, TypedIndex<T2> col);

  // Adds a row to the matrix with the dimension of the given RowType
  TypedIndex<RowType> addRow<RowType>();

  // Adds a col to the matrix with the dimension of the given ColType
  TypedIndex<ColType> addColumn<ColType>();

  void removeRow(TypedIndex<RowType> row);
  void removeColumn(TypedIndex<ColType> col);

  // Copies the blocks between the indices into the dense sub matrix.
  void fillDenseSubMatrix(TypedIndex<T1> topRow, TypedIndex<T2> topColumn, TypedIndex<T3> bottomRow, TypedIndex<T4> bottomColumn, Eigen::Matrix<ScalarType>& denseMatrix);

  // Computes the matrix (non-block) index of the top left part of the matrix block using a TypedIndex<T>. 
  size_t computeRowIndex(TypedIndex<T> idx);
  size_t computeColumnIndex(TypedIndex<T> idx);

  // += operator is overloaded

  // * operator overloaded for Matrix * Row^(T)

};
```
### Sparse Block Row
```cpp
class SparseBlockRow<ScalarType, RowDimension, VariableGroup<V1, V2, ...>> {
  
  // Gets a block vector at a given index.
  MatrixBlock<ScalarType, RowDimension, T1::Dimension> get(TypedIndex<T1> row);

  // Adds a row to the matrix with the dimension of the given RowType
  TypedIndex<ColumnType> addColumn<ColType>();

  void removeColumn(TypedIndex<ColumnType> col);

  // Copies the block matrices between indices into the dense vector. row and column are where 
  // in the dense matrix the row is added to.
  void fillDenseSubMatrix(TypedIndex<T1> leftIndex, rightIndex<T2> bottomIndex, size_t row, size_t column, Eigen::Matrix<ScalarType>& denseMatrix);

  // += operator overloaded

  // * operator overloaded

};
```

