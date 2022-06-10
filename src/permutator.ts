export function permutator(inputArr: any[]) : any[] {
    var results: any[] = [];
  
    function permute(arr: any[], memo: any) : any[] {
      var cur, memo = memo || [];
  
      for (var i = 0; i < arr.length; i++) {
        cur = arr.splice(i, 1);
        if (arr.length === 0) {
          results.push(memo.concat(cur));
        }
        permute(arr.slice(), memo.concat(cur));
        arr.splice(i, 0, cur[0]);
      }
  
      return results;
    }
  
    return permute(inputArr, null);
  }
  
export const permutationIndices3 = permutator([0,1,2])